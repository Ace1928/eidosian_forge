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
def test_read_only_view(self) -> None:
    arr = DataArray(np.random.rand(3, 3), coords={'x': np.arange(3), 'y': np.arange(3)}, dims=('x', 'y'))
    arr = arr.expand_dims({'z': 3}, -1)
    arr['z'] = np.arange(3)
    with pytest.raises(ValueError, match='Do you want to .copy()'):
        arr.loc[0, 0, 0] = 999