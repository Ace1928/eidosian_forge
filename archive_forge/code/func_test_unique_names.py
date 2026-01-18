from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_unique_names(generator_class):
    a = generator_class().random((10, 10), chunks=(5, 5))
    b = generator_class().random((10, 10), chunks=(5, 5))
    assert a.name != b.name