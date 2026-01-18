from __future__ import annotations
import warnings
import pytest
from packaging.version import parse as parse_version
import numpy as np
import dask.array as da
import dask.array.stats
from dask.array.utils import allclose, assert_eq
from dask.delayed import Delayed
def test_skew_raises():
    a = da.ones((7,), chunks=(7,))
    with pytest.raises(ValueError, match='7 samples'):
        dask.array.stats.skewtest(a)