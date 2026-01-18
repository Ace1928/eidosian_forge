from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
def test_reduce_keepdims(self):
    v = Variable(['x', 'y'], self.d)
    assert_identical(v.mean(keepdims=True), Variable(v.dims, np.mean(self.d, keepdims=True)))
    assert_identical(v.mean(dim='x', keepdims=True), Variable(v.dims, np.mean(self.d, axis=0, keepdims=True)))
    assert_identical(v.mean(dim='y', keepdims=True), Variable(v.dims, np.mean(self.d, axis=1, keepdims=True)))
    assert_identical(v.mean(dim=['y', 'x'], keepdims=True), Variable(v.dims, np.mean(self.d, axis=(1, 0), keepdims=True)))
    v = Variable([], 1.0)
    assert_identical(v.mean(keepdims=True), Variable([], np.mean(v.data, keepdims=True)))