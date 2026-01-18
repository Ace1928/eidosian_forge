from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
def test_vlen_dtype() -> None:
    dtype = strings.create_vlen_dtype(str)
    assert dtype.metadata['element_type'] == str
    assert strings.is_unicode_dtype(dtype)
    assert not strings.is_bytes_dtype(dtype)
    assert strings.check_vlen_dtype(dtype) is str
    dtype = strings.create_vlen_dtype(bytes)
    assert dtype.metadata['element_type'] == bytes
    assert not strings.is_unicode_dtype(dtype)
    assert strings.is_bytes_dtype(dtype)
    assert strings.check_vlen_dtype(dtype) is bytes
    dtype = np.dtype('O', metadata={'vlen': str})
    assert strings.check_vlen_dtype(dtype) is str
    assert strings.check_vlen_dtype(np.dtype(object)) is None