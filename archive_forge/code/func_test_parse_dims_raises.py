from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
@pytest.mark.parametrize('dim', [pytest.param('x', id='str_missing'), pytest.param(['a', 'x'], id='list_missing_one'), pytest.param(['x', 2], id='list_missing_all')])
def test_parse_dims_raises(dim) -> None:
    all_dims = ('a', 'b', 1, ('b', 'c'))
    with pytest.raises(ValueError, match="'x'"):
        utils.parse_dims(dim, all_dims, check_exists=True)