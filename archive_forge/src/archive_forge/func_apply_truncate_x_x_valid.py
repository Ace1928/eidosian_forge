from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def apply_truncate_x_x_valid(obj):
    return apply_ufunc(truncate, obj, input_core_dims=[['x']], output_core_dims=[['x']], exclude_dims={'x'})