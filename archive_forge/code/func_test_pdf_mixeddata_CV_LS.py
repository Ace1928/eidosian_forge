import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
@pytest.mark.slow
def test_pdf_mixeddata_CV_LS(self):
    dens_u = nparam.KDEMultivariate(data=[self.c1, self.o, self.o2], var_type='coo', bw='cv_ls')
    npt.assert_allclose(dens_u.bw, [0.70949447, 0.08736727, 0.09220476], atol=1e-06)