from __future__ import annotations
import random
import sys
from copy import deepcopy
from itertools import product
import numpy as np
import pytest
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, ComplexWarning
from dask.array.utils import assert_eq
from dask.base import tokenize
from dask.utils import typename
def test_tokenize_masked_array():
    m = np.ma.masked_array([1, 2, 3], mask=[True, True, False], fill_value=10)
    m2 = np.ma.masked_array([1, 2, 3], mask=[True, True, False], fill_value=0)
    m3 = np.ma.masked_array([1, 2, 3], mask=False, fill_value=10)
    assert tokenize(m) == tokenize(m)
    assert tokenize(m2) == tokenize(m2)
    assert tokenize(m3) == tokenize(m3)
    assert tokenize(m) != tokenize(m2)
    assert tokenize(m) != tokenize(m3)