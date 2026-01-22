import os
import numpy.testing as npt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import statsmodels.sandbox.nonparametric.kernels as kernels
import statsmodels.nonparametric.bandwidths as bandwidths
class TestKdeRefit:
    np.random.seed(12345)
    data1 = np.random.randn(100) * 100
    pdf = KDE(data1)
    pdf.fit()
    data2 = np.random.randn(100) * 100
    pdf2 = KDE(data2)
    pdf2.fit()
    for attr in ['icdf', 'cdf', 'sf']:
        npt.assert_(not np.allclose(getattr(pdf, attr)[:10], getattr(pdf2, attr)[:10]))