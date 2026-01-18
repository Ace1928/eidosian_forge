from __future__ import annotations
import itertools
from typing import Any
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable
from xarray.core import indexing, nputils
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.types import T_Xarray
from xarray.tests import (
def test_stacked_multiindex_min_max(self) -> None:
    data = np.random.randn(3, 23, 4)
    da = DataArray(data, name='value', dims=['replicate', 'rsample', 'exp'], coords=dict(replicate=[0, 1, 2], exp=['a', 'b', 'c', 'd'], rsample=list(range(23))))
    da2 = da.stack(sample=('replicate', 'rsample'))
    s = da2.sample
    assert_array_equal(da2.loc['a', s.max()], data[2, 22, 0])
    assert_array_equal(da2.loc['b', s.min()], data[0, 0, 1])