import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
@pytest.mark.slow
def test_unordered_CV_LS(self):
    dens_ls = nparam.KDEMultivariateConditional(endog=[self.oecd], exog=[self.growth], dep_type='u', indep_type='c', bw='cv_ls')