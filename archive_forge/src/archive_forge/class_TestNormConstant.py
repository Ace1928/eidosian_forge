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
class TestNormConstant:

    def test_norm_constant_calculation(self):
        custom_gauss = kernels.CustomKernel(lambda x: np.exp(-x ** 2 / 2.0))
        gauss_true_const = 0.3989422804014327
        npt.assert_almost_equal(gauss_true_const, custom_gauss.norm_const)