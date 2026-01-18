from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
@pytest.mark.parametrize('numpy_str_type', (np.str_, np.bytes_))
def test_numpy_subclass_handling(numpy_str_type) -> None:
    with pytest.raises(TypeError, match='unsupported type for vlen_dtype'):
        strings.create_vlen_dtype(numpy_str_type)