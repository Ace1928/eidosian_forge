import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
def test_object_to_parametric_internal_error(self):
    object_dtype = type(np.dtype(object))
    other_dtype = type(np.dtype(str))
    cast = get_castingimpl(object_dtype, other_dtype)
    with pytest.raises(TypeError, match='casting from object to the parametric DType'):
        cast._resolve_descriptors((np.dtype('O'), None))