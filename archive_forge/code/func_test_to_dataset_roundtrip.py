from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_to_dataset_roundtrip(self):
    x = self.sp_xr
    assert_equal(x, x.to_dataset('x').to_dataarray('x'))