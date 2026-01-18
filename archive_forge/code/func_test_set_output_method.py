import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('dataframe_lib', ['pandas', 'polars'])
def test_set_output_method(dataframe_lib):
    """Check that the output is a dataframe."""
    lib = pytest.importorskip(dataframe_lib)
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput().fit(X)
    est2 = est.set_output(transform=None)
    assert est2 is est
    X_trans_np = est2.transform(X)
    assert isinstance(X_trans_np, np.ndarray)
    est.set_output(transform=dataframe_lib)
    X_trans_pd = est.transform(X)
    assert isinstance(X_trans_pd, lib.DataFrame)