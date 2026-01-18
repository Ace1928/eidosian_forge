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
@requires_cftime
def test_rename_does_not_change_CFTimeIndex_type(self) -> None:
    time = xr.cftime_range(start='2000', periods=6, freq='2MS', calendar='noleap')
    orig = Dataset(coords={'time': time})
    renamed = orig.rename(time='time_new')
    assert 'time_new' in renamed.xindexes
    assert isinstance(renamed.xindexes['time_new'].to_pandas_index(), CFTimeIndex)
    assert renamed.xindexes['time_new'].to_pandas_index().name == 'time_new'
    assert 'time' in orig.xindexes
    assert isinstance(orig.xindexes['time'].to_pandas_index(), CFTimeIndex)
    assert orig.xindexes['time'].to_pandas_index().name == 'time'
    renamed = orig.rename_dims()
    assert isinstance(renamed.xindexes['time'].to_pandas_index(), CFTimeIndex)
    renamed = orig.rename_vars()
    assert isinstance(renamed.xindexes['time'].to_pandas_index(), CFTimeIndex)