import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
class TestMultiTests2(CheckMultiTestsMixin):

    @classmethod
    def setup_class(cls):
        cls.methods = ['b', 's', 'sh', 'hs', 'h', 'fdr_i', 'fdr_n']
        cls.alpha = 0.05
        cls.res2 = res_multtest2