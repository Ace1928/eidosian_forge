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
def test_unary_ops(self) -> None:
    ds = self.make_example_math_dataset()
    assert_identical(ds.map(abs), abs(ds))
    assert_identical(ds.map(lambda x: x + 4), ds + 4)
    for func in [lambda x: x.isnull(), lambda x: x.round(), lambda x: x.astype(int)]:
        assert_identical(ds.map(func), func(ds))
    assert_identical(ds.isnull(), ~ds.notnull())
    with pytest.raises(AttributeError):
        ds.item
    with pytest.raises(AttributeError):
        ds.searchsorted