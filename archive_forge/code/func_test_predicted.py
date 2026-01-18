import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, assert_equal
from statsmodels.regression.linear_model import GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata
def test_predicted(self):
    res, results = (self.res, self.results)
    assert_allclose(res.fittedvalues, results.fittedvalues, rtol=0.002)
    predicted = res.predict(res.model.exog)
    assert_allclose(predicted, results.fittedvalues, rtol=0.0016)