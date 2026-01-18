from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_parametrized_random_function(generator_class):
    a = generator_class().exponential(1000, (10, 10), chunks=(5, 5))
    assert isinstance(a, Array)
    assert isinstance(a.name, str) and a.name
    assert a.shape == (10, 10)
    assert a.chunks == ((5, 5), (5, 5))
    x = np.array(a)
    assert 10 < x.mean() < 100000
    y = set(x.flat)
    assert len(y) > 90