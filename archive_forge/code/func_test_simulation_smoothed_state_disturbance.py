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
def test_simulation_smoothed_state_disturbance(self):
    assert_allclose(self.sim_a.simulated_state_disturbance, self.sim_b.simulated_state_disturbance, atol=1e-07)