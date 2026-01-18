import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, assert_equal
from statsmodels.regression.linear_model import GLSAR
from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata
def test_rho(self):
    assert_almost_equal(self.res.model.rho, self.results.rho, 3)
    assert_almost_equal(self.res.llf, self.results.ll, 4)