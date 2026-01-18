from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
Check that weighted operations work with unequal coords.


    Parameters
    ----------
    coords_weights : Iterable[Any]
        The coords for the weights.
    coords_data : Iterable[Any]
        The coords for the data.
    expected_value_at_weighted_quantile : float
        The expected value for the quantile of the weighted data.
    