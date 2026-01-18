from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_to_dataset_whole(self) -> None:
    unnamed = DataArray([1, 2], dims='x')
    with pytest.raises(ValueError, match='unable to convert unnamed'):
        unnamed.to_dataset()
    actual = unnamed.to_dataset(name='foo')
    expected = Dataset({'foo': ('x', [1, 2])})
    assert_identical(expected, actual)
    named = DataArray([1, 2], dims='x', name='foo', attrs={'y': 'testattr'})
    actual = named.to_dataset()
    expected = Dataset({'foo': ('x', [1, 2], {'y': 'testattr'})})
    assert_identical(expected, actual)
    actual = named.to_dataset(promote_attrs=True)
    expected = Dataset({'foo': ('x', [1, 2], {'y': 'testattr'})}, attrs={'y': 'testattr'})
    assert_identical(expected, actual)
    with pytest.raises(TypeError):
        actual = named.to_dataset('bar')