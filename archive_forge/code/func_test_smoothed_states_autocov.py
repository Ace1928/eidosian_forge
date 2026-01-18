import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace import dynamic_factor
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from numpy.testing import assert_equal, assert_allclose
def test_smoothed_states_autocov(self):
    assert_allclose(self.results_a.smoothed_state_autocov, self.results_b.smoothed_state_autocov)
    assert_allclose(self.results_a.smoothed_state_autocov[:, :, 0:5], self.augmented_results.smoothed_state_cov[:2, 2:, 1:6], atol=0.0001)
    assert_allclose(self.results_a.smoothed_state_autocov[:, :, 5:-1], self.augmented_results.smoothed_state_cov[:2, 2:, 6:], atol=1e-07)