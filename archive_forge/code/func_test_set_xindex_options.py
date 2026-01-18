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
def test_set_xindex_options(self) -> None:
    ds = Dataset(coords={'foo': ('x', ['a', 'a', 'b', 'b'])})

    class IndexWithOptions(Index):

        def __init__(self, opt):
            self.opt = opt

        @classmethod
        def from_variables(cls, variables, options):
            return cls(options['opt'])
    indexed = ds.set_xindex('foo', IndexWithOptions, opt=1)
    assert getattr(indexed.xindexes['foo'], 'opt') == 1