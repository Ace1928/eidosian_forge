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
def test_unicode_data(self) -> None:
    data = Dataset({'foø': ['ba®']}, attrs={'å': '∑'})
    repr(data)
    byteorder = '<' if sys.byteorder == 'little' else '>'
    expected = dedent('            <xarray.Dataset> Size: 12B\n            Dimensions:  (foø: 1)\n            Coordinates:\n              * foø      (foø) %cU3 12B %r\n            Data variables:\n                *empty*\n            Attributes:\n                å:        ∑' % (byteorder, 'ba®'))
    actual = str(data)
    assert expected == actual