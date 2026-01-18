from statsmodels.compat.pandas import testing as pdt
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
def test_nopatsy(self):
    res = self.res
    data = self.data
    fitted = res.fittedvalues.iloc[1:10:2]
    pred = res.predict(res.model.exog[1:10:2], transform=False)
    assert_allclose(pred, fitted.values, rtol=1e-13)
    x = pd.DataFrame(res.model.exog[1:10:2], index=data.index[1:10:2], columns=res.model.exog_names)
    pred = res.predict(x)
    pdt.assert_index_equal(pred.index, fitted.index)
    assert_allclose(pred.values, fitted.values, rtol=1e-13)
    pred = res.predict(res.model.exog[1], transform=False)
    assert_allclose(pred, fitted.values[0], rtol=1e-13)
    pred = res.predict(x.iloc[0])
    pdt.assert_index_equal(pred.index, fitted.index[:1])
    assert_allclose(pred.values[0], fitted.values[0], rtol=1e-13)