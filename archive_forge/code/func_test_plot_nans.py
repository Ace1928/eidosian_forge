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
def test_plot_nans(self) -> None:
    x1 = self.darray[:5]
    x2 = self.darray.copy()
    x2[5:] = np.nan
    clim1 = self.plotfunc(x1).get_clim()
    clim2 = self.plotfunc(x2).get_clim()
    assert clim1 == clim2