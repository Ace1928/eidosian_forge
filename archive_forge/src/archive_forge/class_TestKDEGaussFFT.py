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
class TestKDEGaussFFT(CheckKDE):

    @classmethod
    def setup_class(cls):
        cls.decimal_density = 2
        res1 = KDE(Xi)
        res1.fit(kernel='gau', fft=True, bw='silverman')
        cls.res1 = res1
        rfname2 = os.path.join(curdir, 'results', 'results_kde_fft.csv')
        cls.res_density = np.genfromtxt(open(rfname2, 'rb'))