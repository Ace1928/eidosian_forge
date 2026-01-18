import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
@pytest.mark.skipif(IS_WASM, reason='no threading')
def test_structured_advanced_indexing(self):
    from concurrent.futures import ThreadPoolExecutor
    dt = np.dtype([('', 'f8')])
    dt = np.dtype([('', dt)] * 2)
    dt = np.dtype([('', dt)] * 2)
    arr = np.random.uniform(size=(6000, 8)).view(dt)[:, 0]
    rng = np.random.default_rng()

    def func(arr):
        indx = rng.integers(0, len(arr), size=6000, dtype=np.intp)
        arr[indx]
    tpe = ThreadPoolExecutor(max_workers=8)
    futures = [tpe.submit(func, arr) for _ in range(10)]
    for f in futures:
        f.result()
    assert arr.dtype is dt