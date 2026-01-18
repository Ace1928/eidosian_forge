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
def test_plot_size(self) -> None:
    self.darray[:, 0, 0].plot(figsize=(13, 5))
    assert tuple(plt.gcf().get_size_inches()) == (13, 5)
    self.darray.plot(figsize=(13, 5))
    assert tuple(plt.gcf().get_size_inches()) == (13, 5)
    self.darray.plot(size=5)
    assert plt.gcf().get_size_inches()[1] == 5
    self.darray.plot(size=5, aspect=2)
    assert tuple(plt.gcf().get_size_inches()) == (10, 5)
    with pytest.raises(ValueError, match='cannot provide both'):
        self.darray.plot(ax=plt.gca(), figsize=(3, 4))
    with pytest.raises(ValueError, match='cannot provide both'):
        self.darray.plot(size=5, figsize=(3, 4))
    with pytest.raises(ValueError, match='cannot provide both'):
        self.darray.plot(size=5, ax=plt.gca())
    with pytest.raises(ValueError, match='cannot provide `aspect`'):
        self.darray.plot(aspect=1)