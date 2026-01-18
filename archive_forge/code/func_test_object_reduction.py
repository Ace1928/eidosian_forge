from __future__ import annotations
import os
import warnings
from contextlib import nullcontext as does_not_warn
from itertools import permutations, zip_longest
import pytest
import itertools
import dask.array as da
import dask.config as config
from dask.array.numpy_compat import NUMPY_GE_122, ComplexWarning
from dask.array.utils import assert_eq, same_keys
from dask.core import get_deps
@pytest.mark.parametrize('method', ['sum', 'mean', 'prod'])
def test_object_reduction(method):
    arr = da.ones(1).astype(object)
    result = getattr(arr, method)().compute()
    assert result == 1