from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
@pytest.mark.parametrize('dim', [pytest.param(None, id='None'), pytest.param(..., id='ellipsis')])
def test_parse_dims_replace_none(dim: None | ellipsis) -> None:
    all_dims = ('a', 'b', 1, ('b', 'c'))
    actual = utils.parse_dims(dim, all_dims, replace_none=True)
    assert actual == all_dims