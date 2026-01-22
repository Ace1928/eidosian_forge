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
@pytest.mark.filterwarnings('ignore:tight_layout cannot')
class TestFacetedLinePlotsLegend(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.darray = xr.tutorial.scatter_example_dataset()

    def test_legend_labels(self) -> None:
        fg = self.darray.A.plot.line(col='x', row='w', hue='z')
        all_legend_labels = [t.get_text() for t in fg.figlegend.texts]
        assert sorted(all_legend_labels) == ['0', '1', '2', '3']