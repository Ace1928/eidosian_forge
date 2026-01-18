import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize(('from_dt', 'to_dt', 'expected_off'), [('i', '(1,1)i', 0), ('(1,1)i', 'i', 0), ('(2,1)i', '(2,1)i', 0), ('i', dict(names=['a'], formats=['i'], offsets=[2]), None), (dict(names=['a'], formats=['i'], offsets=[2]), 'i', 2), ('i', dict(names=['a', 'b'], formats=['i', 'i'], offsets=[2, 2]), None), ('i,i', 'i,i,i', None), ('i4', 'V3', 0), ('i4', 'V4', 0), ('i4', 'V10', None), ('O', 'V4', None), ('O', 'V8', None), ('V4', 'V3', 0), ('V4', 'V4', 0), ('V3', 'V4', None), ('V4', 'i4', None), ('i,i', 'i,i,i', None)])
def test_structured_view_offsets_paramteric(self, from_dt, to_dt, expected_off):
    from_dt = np.dtype(from_dt)
    to_dt = np.dtype(to_dt)
    cast = get_castingimpl(type(from_dt), type(to_dt))
    _, _, view_off = cast._resolve_descriptors((from_dt, to_dt))
    assert view_off == expected_off