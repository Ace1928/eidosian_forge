import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
class TestTTPowerTwoS2(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        res2.n = 20
        res2.d = 0.1
        res2.sig_level = 0.05
        res2.power = 0.06095912465411235
        res2.alternative = 'two.sided'
        res2.note = 'n is number in *each* group'
        res2.method = 'Two-sample t test power calculation'
        cls.res2 = res2
        cls.kwds = {'effect_size': res2.d, 'nobs1': res2.n, 'alpha': res2.sig_level, 'power': res2.power, 'ratio': 1}
        cls.kwds_extra = {}
        cls.cls = smp.TTestIndPower