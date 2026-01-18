from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
@requires_pandas
def test_pandas_multiindex():
    index = pd.MultiIndex.from_product([range(50), list('abcdefghilmnopqrstuvwxyz')])
    actual_size = sys.getsizeof(index)
    assert 0.5 * actual_size < sizeof(index) < 3 * actual_size
    assert isinstance(sizeof(index), int)