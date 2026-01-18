from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_replace_compiled_regex(dtype) -> None:
    values = xr.DataArray(['fooBAD__barBAD'], dims=['x']).astype(dtype)
    pat = re.compile(dtype('BAD[_]*'))
    result = values.str.replace(pat, '')
    expected = xr.DataArray(['foobar'], dims=['x']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.replace(pat, '', n=1)
    expected = xr.DataArray(['foobarBAD'], dims=['x']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    pat2 = xr.DataArray([re.compile(dtype('BAD[_]*')), re.compile(dtype('AD[_]*'))], dims=['y'])
    result = values.str.replace(pat2, '')
    expected = xr.DataArray([['foobar', 'fooBbarB']], dims=['x', 'y']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    repl = xr.DataArray(['', 'spam'], dims=['y']).astype(dtype)
    result = values.str.replace(pat2, repl, n=1)
    expected = xr.DataArray([['foobarBAD', 'fooBspambarBAD']], dims=['x', 'y']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    values = xr.DataArray(['fooBAD__barBAD__bad']).astype(dtype)
    pat3 = re.compile(dtype('BAD[_]*'))
    with pytest.raises(ValueError, match='Flags cannot be set when pat is a compiled regex.'):
        result = values.str.replace(pat3, '', flags=re.IGNORECASE)
    with pytest.raises(ValueError, match='Case cannot be set when pat is a compiled regex.'):
        result = values.str.replace(pat3, '', case=False)
    with pytest.raises(ValueError, match='Case cannot be set when pat is a compiled regex.'):
        result = values.str.replace(pat3, '', case=True)
    values = xr.DataArray(['fooBAD__barBAD']).astype(dtype)
    repl2 = lambda m: m.group(0).swapcase()
    pat4 = re.compile(dtype('[a-z][A-Z]{2}'))
    result = values.str.replace(pat4, repl2, n=2)
    expected = xr.DataArray(['foObaD__baRbaD']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)