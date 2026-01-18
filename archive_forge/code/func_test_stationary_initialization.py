import copy
import pickle
import numpy as np
import pandas as pd
import os
import pytest
from scipy.linalg.blas import find_best_blas_type
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace import _representation, _kalman_filter
from .results import results_kalman_filter
from numpy.testing import assert_almost_equal, assert_allclose
def test_stationary_initialization():
    check_stationary_initialization_1dim(np.float32)
    check_stationary_initialization_1dim(np.float64)
    check_stationary_initialization_1dim(np.complex64)
    check_stationary_initialization_1dim(np.complex128)
    check_stationary_initialization_2dim(np.float32)
    check_stationary_initialization_2dim(np.float64)
    check_stationary_initialization_2dim(np.complex64)
    check_stationary_initialization_2dim(np.complex128)