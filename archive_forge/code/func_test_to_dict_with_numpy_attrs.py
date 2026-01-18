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
def test_to_dict_with_numpy_attrs(self) -> None:
    x = np.random.randn(10)
    y = np.random.randn(10)
    t = list('abcdefghij')
    attrs = {'created': np.float64(1998), 'coords': np.array([37, -110.1, 100]), 'maintainer': 'bar'}
    ds = Dataset({'a': ('t', x, attrs), 'b': ('t', y, attrs), 't': ('t', t)})
    expected_attrs = {'created': attrs['created'].item(), 'coords': attrs['coords'].tolist(), 'maintainer': 'bar'}
    actual = ds.to_dict()
    assert expected_attrs == actual['data_vars']['a']['attrs']