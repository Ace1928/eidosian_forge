from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
@requires_pandas
def test_pandas_contiguous_dtypes():
    """2+ contiguous columns of the same dtype in the same DataFrame share the same
    surface thus have lower overhead
    """
    df1 = pd.DataFrame([[1, 2.2], [3, 4.4]])
    df2 = pd.DataFrame([[1.1, 2.2], [3.3, 4.4]])
    assert sizeof(df2) < sizeof(df1)