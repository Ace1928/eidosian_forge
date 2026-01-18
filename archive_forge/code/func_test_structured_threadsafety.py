import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have threads")
def test_structured_threadsafety(self):
    from concurrent.futures import ThreadPoolExecutor
    dt = np.dtype([('', 'f8')])
    dt = np.dtype([('', dt)])
    dt = np.dtype([('', dt)] * 2)
    arr = np.random.uniform(size=(5000, 4)).view(dt)[:, 0]

    def func(arr):
        arr.nonzero()
    tpe = ThreadPoolExecutor(max_workers=8)
    futures = [tpe.submit(func, arr) for _ in range(10)]
    for f in futures:
        f.result()
    assert arr.dtype is dt