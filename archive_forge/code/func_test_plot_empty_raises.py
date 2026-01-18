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
@requires_matplotlib
@pytest.mark.parametrize('val', [pytest.param([], id='empty'), pytest.param(0, id='scalar')])
@pytest.mark.parametrize('method', ['__call__', 'line', 'step', 'contour', 'contourf', 'hist', 'imshow', 'pcolormesh', 'scatter', 'surface'])
def test_plot_empty_raises(val: list | float, method: str) -> None:
    da = xr.DataArray(val)
    with pytest.raises(TypeError, match='No numeric data'):
        getattr(da.plot, method)()