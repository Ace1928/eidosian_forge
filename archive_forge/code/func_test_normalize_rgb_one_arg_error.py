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
def test_normalize_rgb_one_arg_error(self) -> None:
    da = DataArray(easy_array((5, 5, 3), start=-0.6, stop=1.4))
    for vmin, vmax in ((None, -1), (2, None)):
        with pytest.raises(ValueError):
            da.plot.imshow(vmin=vmin, vmax=vmax)
    for vmin2, vmax2 in ((-1.2, -1), (2, 2.1)):
        da.plot.imshow(vmin=vmin2, vmax=vmax2)