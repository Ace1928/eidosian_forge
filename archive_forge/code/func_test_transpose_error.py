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
def test_transpose_error() -> None:
    ds = xr.Dataset({'foo': (('x', 'y'), [[21]]), 'bar': (('x', 'y'), [[12]])})
    with pytest.raises(TypeError, match=re.escape("transpose requires dims to be passed as multiple arguments. Expected `'y', 'x'`. Received `['y', 'x']` instead")):
        ds.transpose(['y', 'x'])