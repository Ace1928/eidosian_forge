import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
class TestFtestPowerF2(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        res2.u = 5
        res2.v = 19
        res2.f2 = 0.09
        res2.sig_level = 0.1
        res2.power = 0.235454222377575
        res2.method = 'Multiple regression power calculation'
        cls.res2 = res2
        cls.kwds = {'effect_size': res2.f2, 'df_num': res2.u, 'df_denom': res2.v, 'alpha': res2.sig_level, 'power': res2.power}
        cls.kwds_extra = {}
        cls.args_names = ['effect_size', 'df_num', 'df_denom', 'alpha']
        cls.cls = smp.FTestPowerF2
        cls.decimal = 5