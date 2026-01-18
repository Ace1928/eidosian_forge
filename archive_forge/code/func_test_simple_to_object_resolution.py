import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('dtype', simple_dtype_instances())
def test_simple_to_object_resolution(self, dtype):
    object_dtype = type(np.dtype(object))
    cast = get_castingimpl(type(dtype), object_dtype)
    safety, (_, res_dt), view_off = cast._resolve_descriptors((dtype, None))
    assert safety == Casting.safe
    assert view_off is None
    assert res_dt is np.dtype('O')