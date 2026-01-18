from statsmodels.compat.pandas import testing as pdt
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
def test_predict_offset(self):
    res = self.res
    data = self.data
    fitted = res.fittedvalues.iloc[1:10:2]
    offset = np.arange(len(fitted))
    fitted = fitted + offset
    pred = res.predict(data.iloc[1:10:2], offset=offset)
    pdt.assert_index_equal(pred.index, fitted.index)
    assert_allclose(pred.values, fitted.values, rtol=1e-13)
    xd = dict(zip(data.columns, data.iloc[1:10:2].values.T))
    pred = res.predict(xd, offset=offset)
    assert_equal(pred.index, np.arange(len(pred)))
    assert_allclose(pred.values, fitted.values, rtol=1e-13)
    data2 = data.iloc[1:10:2].copy()
    data2['offset'] = offset
    pred = res.predict(data2, offset=data2['offset'])
    pdt.assert_index_equal(pred.index, fitted.index)
    assert_allclose(pred.values, fitted.values, rtol=1e-13)
    data2 = data.iloc[1:10:2].copy()
    data2['offset'] = offset
    data2.iloc[0, 1] = np.nan
    pred = res.predict(data2, offset=data2['offset'])
    pdt.assert_index_equal(pred.index, fitted.index)
    fitted_nan = fitted.copy()
    fitted_nan.iloc[0] = np.nan
    assert_allclose(pred.values, fitted_nan.values, rtol=1e-13)