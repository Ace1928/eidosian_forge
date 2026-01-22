import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
class TestTTPowerOneS4(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        res2.n = 20
        res2.d = 0.05
        res2.sig_level = 0.05
        res2.power = 0.0764888785042198
        res2.alternative = 'greater'
        res2.note = 'NULL'
        res2.method = 'One-sample t test power calculation'
        cls.res2 = res2
        cls.kwds = {'effect_size': res2.d, 'nobs': res2.n, 'alpha': res2.sig_level, 'power': res2.power}
        cls.kwds_extra = {'alternative': 'larger'}
        cls.cls = smp.TTestPower