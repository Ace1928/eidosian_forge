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
def test_unnamed_args(self) -> None:
    g = self.darray.plot.line('o--', row='row', col='col', hue='hue')
    lines = [q for q in g.axs.flat[0].get_children() if isinstance(q, mpl.lines.Line2D)]
    assert lines[0].get_marker() == 'o'
    assert lines[0].get_linestyle() == '--'