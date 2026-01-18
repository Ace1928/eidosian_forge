import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.regression._prediction import get_prediction
def test_predict_remove_data():
    endog = [i + np.random.normal(scale=0.1) for i in range(100)]
    exog = [i for i in range(100)]
    model = WLS(endog, exog, weights=[1 for _ in range(100)]).fit()
    model.scale
    model.remove_data()
    scalar = model.get_prediction(1).predicted_mean
    pred = model.get_prediction([1])
    one_d = pred.predicted_mean
    assert_allclose(scalar, one_d)
    pred.summary_frame()
    series = model.get_prediction(pd.Series([1])).predicted_mean
    assert_allclose(scalar, series)