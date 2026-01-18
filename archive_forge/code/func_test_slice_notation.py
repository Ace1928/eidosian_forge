import os
import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother
from statsmodels.tsa.statespace import tools, sarimax
from .results import results_kalman_filter
from numpy.testing import (
def test_slice_notation():
    endog = np.arange(10) * 1.0
    mod = KalmanFilter(k_endog=1, k_states=2)
    mod.bind(endog)

    def set_designs():
        mod['designs'] = 1

    def set_designs2():
        mod['designs', 0, 0] = 1

    def set_designs3():
        mod[0] = 1
    assert_raises(IndexError, set_designs)
    assert_raises(IndexError, set_designs2)
    assert_raises(IndexError, set_designs3)
    assert_raises(IndexError, lambda: mod['designs'])
    assert_raises(IndexError, lambda: mod['designs', 0, 0, 0])
    assert_raises(IndexError, lambda: mod[0])
    assert_equal(mod.design[0, 0, 0], 0)
    mod['design', 0, 0, 0] = 1
    assert_equal(mod['design'].sum(), 1)
    assert_equal(mod.design[0, 0, 0], 1)
    assert_equal(mod['design', 0, 0, 0], 1)
    mod['design'] = np.zeros(mod['design'].shape)
    assert_equal(mod.design[0, 0], 0)
    mod['design', 0, 0] = 1
    assert_equal(mod.design[0, 0], 1)
    assert_equal(mod['design', 0, 0], 1)