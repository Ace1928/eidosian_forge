import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
class TestVARAutocovariancesClassicalSmoothing(TestVARAutocovariances):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, smooth_method=SMOOTH_CLASSICAL, **kwargs)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, SMOOTH_CLASSICAL)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, SMOOTH_CLASSICAL)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_CLASSICAL)