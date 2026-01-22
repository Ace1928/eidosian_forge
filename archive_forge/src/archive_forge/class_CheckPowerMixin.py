import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
class CheckPowerMixin:

    def test_power(self):
        kwds = copy.copy(self.kwds)
        del kwds['power']
        kwds.update(self.kwds_extra)
        if hasattr(self, 'decimal'):
            decimal = self.decimal
        else:
            decimal = 6
        res1 = self.cls()
        assert_almost_equal(res1.power(**kwds), self.res2.power, decimal=decimal)

    def test_positional(self):
        res1 = self.cls()
        kwds = copy.copy(self.kwds)
        del kwds['power']
        kwds.update(self.kwds_extra)
        if hasattr(self, 'args_names'):
            args_names = self.args_names
        else:
            nobs_ = 'nobs' if 'nobs' in kwds else 'nobs1'
            args_names = ['effect_size', nobs_, 'alpha']
        args = [kwds.pop(arg) for arg in args_names]
        if hasattr(self, 'decimal'):
            decimal = self.decimal
        else:
            decimal = 6
        res = res1.power(*args, **kwds)
        assert_almost_equal(res, self.res2.power, decimal=decimal)

    def test_roots(self):
        kwds = copy.copy(self.kwds)
        kwds.update(self.kwds_extra)
        for key in self.kwds:
            value = kwds[key]
            kwds[key] = None
            result = self.cls().solve_power(**kwds)
            assert_allclose(result, value, rtol=0.001, err_msg=key + ' failed')
            kwds[key] = value

    @pytest.mark.matplotlib
    def test_power_plot(self, close_figures):
        if self.cls in [smp.FTestPower, smp.FTestPowerF2]:
            pytest.skip('skip FTestPower plot_power')
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        fig = self.cls().plot_power(dep_var='nobs', nobs=np.arange(2, 100), effect_size=np.array([0.1, 0.2, 0.3, 0.5, 1]), ax=ax, title='Power of t-Test', **self.kwds_extra)
        ax = fig.add_subplot(2, 1, 2)
        self.cls().plot_power(dep_var='es', nobs=np.array([10, 20, 30, 50, 70, 100]), effect_size=np.linspace(0.01, 2, 51), ax=ax, title='', **self.kwds_extra)