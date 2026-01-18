from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_equal, assert_allclose
from . import kfas_helpers
def test_scaled_smoothed_diffuse2_estimator_cov(self, rtol_diffuse=None):
    actual = self.results_a.scaled_smoothed_diffuse2_estimator_cov
    desired = self.results_b.scaled_smoothed_diffuse2_estimator_cov
    self.check_object(actual, desired, rtol_diffuse)