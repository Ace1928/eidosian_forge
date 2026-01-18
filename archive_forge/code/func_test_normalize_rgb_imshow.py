from __future__ import annotations
import contextlib
import inspect
import math
from collections.abc import Hashable
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.plot as xplt
from xarray import DataArray, Dataset
from xarray.namedarray.utils import module_available
from xarray.plot.dataarray_plot import _infer_interval_breaks
from xarray.plot.dataset_plot import _infer_meta_data
from xarray.plot.utils import (
from xarray.tests import (
@pytest.mark.parametrize(['vmin', 'vmax', 'robust'], [(-1, None, False), (None, 2, False), (-1, 1, False), (0, 0, False), (0, None, True), (None, -1, True)])
def test_normalize_rgb_imshow(self, vmin: float | None, vmax: float | None, robust: bool) -> None:
    da = DataArray(easy_array((5, 5, 3), start=-0.6, stop=1.4))
    arr = da.plot.imshow(vmin=vmin, vmax=vmax, robust=robust).get_array()
    assert arr is not None
    assert 0 <= arr.min() <= arr.max() <= 1