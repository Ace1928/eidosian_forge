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
@pytest.mark.slow
def test_robust(self) -> None:
    z = np.zeros((20, 20, 2))
    darray = DataArray(z, dims=['y', 'x', 'z'])
    darray[:, :, 1] = 1
    darray[2, 0, 0] = -1000
    darray[3, 0, 0] = 1000
    g = xplt.FacetGrid(darray, col='z')
    g.map_dataarray(xplt.imshow, 'x', 'y', robust=True)
    numbers = set()
    alltxt = text_in_fig()
    for txt in alltxt:
        try:
            numbers.add(float(txt))
        except ValueError:
            pass
    largest = max((abs(x) for x in numbers))
    assert largest < 21