from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_to_and_from_dict_with_time_dim(self) -> None:
    x = np.random.randn(10, 3)
    y = np.random.randn(10, 3)
    t = pd.date_range('20130101', periods=10)
    lat = [77.7, 83.2, 76]
    ds = Dataset({'a': (['t', 'lat'], x), 'b': (['t', 'lat'], y), 't': ('t', t), 'lat': ('lat', lat)})
    roundtripped = Dataset.from_dict(ds.to_dict())
    assert_identical(ds, roundtripped)