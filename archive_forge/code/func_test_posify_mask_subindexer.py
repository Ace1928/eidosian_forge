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
@pytest.mark.parametrize('indices, expected', [(np.arange(5), np.arange(5)), (np.array([0, -1, -1]), np.array([0, 0, 0])), (np.array([-1, 1, -1]), np.array([1, 1, 1])), (np.array([-1, -1, 2]), np.array([2, 2, 2])), (np.array([-1]), np.array([0])), (np.array([0, -1, 1, -1, -1]), np.array([0, 0, 1, 1, 1])), (np.array([0, -1, -1, -1, 1]), np.array([0, 0, 0, 0, 1]))])
def test_posify_mask_subindexer(indices, expected) -> None:
    actual = indexing._posify_mask_subindexer(indices)
    np.testing.assert_array_equal(expected, actual)