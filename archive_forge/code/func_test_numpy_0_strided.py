from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
def test_numpy_0_strided():
    np = pytest.importorskip('numpy')
    x = np.broadcast_to(1, (100, 100, 100))
    assert sizeof(x) <= 8