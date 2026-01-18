from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
def test_where_dispatching(self):
    a = np.arange(10)
    b = a > 3
    x = da.from_array(a, 5)
    y = da.from_array(b, 5)
    expected = DataArray(a).where(b)
    self.assertLazyAndEqual(expected, DataArray(a).where(y))
    self.assertLazyAndEqual(expected, DataArray(x).where(b))
    self.assertLazyAndEqual(expected, DataArray(x).where(y))