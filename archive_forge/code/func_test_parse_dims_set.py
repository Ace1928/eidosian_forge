from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
def test_parse_dims_set() -> None:
    all_dims = ('a', 'b', 1, ('b', 'c'))
    dim = {'a', 1}
    actual = utils.parse_dims(dim, all_dims)
    assert set(actual) == dim