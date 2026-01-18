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
def test_verbose_facetgrid(self) -> None:
    a = easy_array((10, 15, 3))
    d = DataArray(a, dims=['y', 'x', 'z'])
    g = xplt.FacetGrid(d, col='z', subplot_kws=self.subplot_kws)
    g.map_dataarray(self.plotfunc, 'x', 'y')
    for ax in g.axs.flat:
        assert ax.has_data()