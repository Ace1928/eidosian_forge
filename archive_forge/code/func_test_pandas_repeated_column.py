from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
@requires_pandas
def test_pandas_repeated_column():
    df = pd.DataFrame({'x': list(range(10000))})
    df2 = df[['x', 'x', 'x']]
    df3 = pd.DataFrame({'x': list(range(10000)), 'y': list(range(10000))})
    assert 80000 < sizeof(df) < 85000
    assert 80000 < sizeof(df2) < 85000
    assert 160000 < sizeof(df3) < 165000