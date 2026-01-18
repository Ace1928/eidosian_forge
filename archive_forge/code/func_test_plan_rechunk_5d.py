from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_plan_rechunk_5d():
    c = (10,) * 1
    f = (1,) * 10
    steps = _plan((c, c, c, c, c), (f, f, f, f, f))
    _assert_steps(steps, [(f, f, f, f, f)])
    steps = _plan((f, f, f, f, c), (c, c, c, f, f))
    _assert_steps(steps, [(c, c, c, f, c), (c, c, c, f, f)])
    steps = _plan((c, c, f, f, c), (c, c, c, f, f), block_size_limit=20000.0)
    _assert_steps(steps, [(c, c, c, f, c), (c, c, c, f, f)])