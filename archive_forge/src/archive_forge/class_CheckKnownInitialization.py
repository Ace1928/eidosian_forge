import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class CheckKnownInitialization:

    @classmethod
    def setup_class(cls, mod, start_params):
        cls.mod = mod
        cls.start_params = start_params
        endog = mod.data.orig_endog
        cls.res = cls.mod.fit(start_params, disp=0, maxiter=100)
        cls.initial_level = cls.res.params.get('initial_level', None)
        cls.initial_trend = cls.res.params.get('initial_trend', None)
        cls.initial_seasonal = None
        if cls.mod.seasonal:
            cls.initial_seasonal = [cls.res.params['initial_seasonal']] + [cls.res.params['initial_seasonal.L%d' % i] for i in range(1, cls.mod.seasonal_periods - 1)]
        cls.params = cls.res.params[:'initial_level'].drop('initial_level')
        cls.init_params = cls.res.params['initial_level':]
        cls.known_mod = cls.mod.clone(endog, initialization_method='known', initial_level=cls.initial_level, initial_trend=cls.initial_trend, initial_seasonal=cls.initial_seasonal)

    def test_given_params(self):
        known_res = self.known_mod.filter(self.params)
        assert_allclose(known_res.llf, self.res.llf)
        assert_allclose(known_res.predicted_state, self.res.predicted_state)
        assert_allclose(known_res.predicted_state_cov, self.res.predicted_state_cov)
        assert_allclose(known_res.filtered_state, self.res.filtered_state)

    def test_estimated_params(self):
        fit_res1 = self.mod.fit_constrained(self.init_params.to_dict(), start_params=self.start_params, includes_fixed=True, disp=0)
        fit_res2 = self.known_mod.fit(self.start_params[:'initial_level'].drop('initial_level'), disp=0)
        assert_allclose(fit_res1.params[:'initial_level'].drop('initial_level'), fit_res2.params)
        assert_allclose(fit_res1.llf, fit_res2.llf)
        assert_allclose(fit_res1.scale, fit_res2.scale)
        assert_allclose(fit_res1.predicted_state, fit_res2.predicted_state)
        assert_allclose(fit_res1.predicted_state_cov, fit_res2.predicted_state_cov)
        assert_allclose(fit_res1.filtered_state, fit_res2.filtered_state)