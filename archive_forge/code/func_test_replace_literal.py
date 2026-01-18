from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_replace_literal(dtype) -> None:
    values = xr.DataArray(['f.o', 'foo'], dims=['X']).astype(dtype)
    expected = xr.DataArray(['bao', 'bao'], dims=['X']).astype(dtype)
    result = values.str.replace('f.', 'ba')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = xr.DataArray(['bao', 'foo'], dims=['X']).astype(dtype)
    result = values.str.replace('f.', 'ba', regex=False)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    pat = xr.DataArray(['f.', '.o'], dims=['yy']).astype(dtype)
    expected = xr.DataArray([['bao', 'fba'], ['bao', 'bao']], dims=['X', 'yy']).astype(dtype)
    result = values.str.replace(pat, 'ba')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = xr.DataArray([['bao', 'fba'], ['foo', 'foo']], dims=['X', 'yy']).astype(dtype)
    result = values.str.replace(pat, 'ba', regex=False)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    callable_repl = lambda m: m.group(0).swapcase()
    compiled_pat = re.compile('[a-z][A-Z]{2}')
    msg = 'Cannot use a callable replacement when regex=False'
    with pytest.raises(ValueError, match=msg):
        values.str.replace('abc', callable_repl, regex=False)
    msg = 'Cannot use a compiled regex as replacement pattern with regex=False'
    with pytest.raises(ValueError, match=msg):
        values.str.replace(compiled_pat, '', regex=False)