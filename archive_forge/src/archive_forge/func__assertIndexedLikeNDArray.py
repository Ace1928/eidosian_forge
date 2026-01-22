from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
def _assertIndexedLikeNDArray(self, variable, expected_value0, expected_dtype=None):
    """Given a 1-dimensional variable, verify that the variable is indexed
        like a numpy.ndarray.
        """
    assert variable[0].shape == ()
    assert variable[0].ndim == 0
    assert variable[0].size == 1
    assert variable.equals(variable.copy())
    assert variable.identical(variable.copy())
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', "In the future, 'NAT == x'")
        np.testing.assert_equal(variable.values[0], expected_value0)
        np.testing.assert_equal(variable[0].values, expected_value0)
    if expected_dtype is None:
        assert type(variable.values[0]) == type(expected_value0)
        assert type(variable[0].values) == type(expected_value0)
    elif expected_dtype is not False:
        assert variable.values[0].dtype == expected_dtype
        assert variable[0].values.dtype == expected_dtype