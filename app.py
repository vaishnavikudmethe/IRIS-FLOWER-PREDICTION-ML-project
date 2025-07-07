from flask import Flask,jsonify ,request,render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("ML_Model.pkl")

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict',methods=["POST"])
def predict():
	if request.method == 'POST':
        
        #Getting data from UI
	    sepal_length = request.form['sepal_length']
	    sepal_width = request.form['sepal_width']
	    petal_length = request.form['petal_length']
	    petal_width = request.form['petal_width']

	    #Optional Code
	    print("Values from UI :")
	    print("Sepal Length :- " ,sepal_length)
	    print("Sepal Width :- ",sepal_width)
	    print("Petal Length :- " ,petal_length)
	    print("Petal Width :- ",petal_width)
	    print("Sepal Length :- " ,type(sepal_length))

	    #Type Cast into Float
	    SL = float(sepal_length)
	    SW = float(sepal_width)
	    PL = float(petal_length)
	    PW = float(petal_width)

	    print("Sepal Length :- " ,type(SL))

	    result = model.predict([[SL,SW,PL,PW]])[0]

	    print("Result :- " ,result)

	    
	    # return jsonify({"Prediction is - ":result})
	return render_template("index.html", result=result)

if __name__ =='__main__':
	# app.run(debug=True)
	app.run(debug=True, port=5050)
