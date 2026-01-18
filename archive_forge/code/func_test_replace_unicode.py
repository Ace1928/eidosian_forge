from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_replace_unicode() -> None:
    values = xr.DataArray([b'abcd,\xc3\xa0'.decode('utf-8')])
    expected = xr.DataArray([b'abcd, \xc3\xa0'.decode('utf-8')])
    pat = re.compile('(?<=\\w),(?=\\w)', flags=re.UNICODE)
    result = values.str.replace(pat, ', ')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    values = xr.DataArray([b'abcd,\xc3\xa0'.decode('utf-8')], dims=['X'])
    expected = xr.DataArray([[b'abcd, \xc3\xa0'.decode('utf-8'), b'BAcd,\xc3\xa0'.decode('utf-8')]], dims=['X', 'Y'])
    pat2 = xr.DataArray([re.compile('(?<=\\w),(?=\\w)', flags=re.UNICODE), 'ab'], dims=['Y'])
    repl = xr.DataArray([', ', 'BA'], dims=['Y'])
    result = values.str.replace(pat2, repl)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)