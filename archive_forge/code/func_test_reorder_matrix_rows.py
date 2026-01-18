import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
def test_reorder_matrix_rows():
    nobs = 5
    k_endog = 3
    k_states = 3
    missing = np.zeros((k_endog, nobs))
    given = np.zeros((k_endog, k_states, nobs))
    given[:, :, :] = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])[:, :, np.newaxis]
    desired = given.copy()
    missing[0, 0] = 1
    given[:, :, 0] = np.array([[21, 22, 23], [31, 32, 33], [0, 0, 0]])
    desired[0, :, 0] = 0
    missing[:2, 1] = 1
    given[:, :, 1] = np.array([[31, 32, 33], [0, 0, 0], [0, 0, 0]])
    desired[:2, :, 1] = 0
    missing[0, 2] = 1
    missing[2, 2] = 1
    given[:, :, 2] = np.array([[21, 22, 23], [0, 0, 0], [0, 0, 0]])
    desired[0, :, 2] = 0
    desired[2, :, 2] = 0
    missing[1, 3] = 1
    given[:, :, 3] = np.array([[11, 12, 13], [31, 32, 33], [0, 0, 0]])
    desired[1, :, 3] = 0
    missing[2, 4] = 1
    given[:, :, 4] = np.array([[11, 12, 13], [21, 22, 23], [0, 0, 0]])
    desired[2, :, 4] = 0
    actual = np.asfortranarray(given)
    missing = np.asfortranarray(missing.astype(np.int32))
    tools.reorder_missing_matrix(actual, missing, True, False, False, inplace=True)
    assert_equal(actual, desired)