import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_set_output_named_tuple_out():
    """Check that namedtuples are kept by default."""
    Output = namedtuple('Output', 'X, Y')
    X = np.asarray([[1, 2, 3]])
    est = EstimatorReturnTuple(OutputTuple=Output)
    X_trans = est.transform(X)
    assert isinstance(X_trans, Output)
    assert_array_equal(X_trans.X, X)
    assert_array_equal(X_trans.Y, 2 * X)