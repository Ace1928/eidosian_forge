from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
def zip_mappings(*mappings):
    for key in set(mappings[0]).intersection(*mappings[1:]):
        yield (key, tuple((m[key] for m in mappings)))