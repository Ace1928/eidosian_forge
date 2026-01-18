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
def test_set_output_list_input(dataframe_lib):
    """Check set_output for list input.

    Non-regression test for #27037.
    """
    lib = pytest.importorskip(dataframe_lib)
    X = [[0, 1, 2, 3], [4, 5, 6, 7]]
    est = EstimatorWithListInput()
    est.set_output(transform=dataframe_lib)
    X_out = est.fit(X).transform(X)
    assert isinstance(X_out, lib.DataFrame)
    assert_array_equal(X_out.columns, ['X0', 'X1', 'X2', 'X3'])