import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('other_dt', ['S8', '<U8', '>U8'])
@pytest.mark.parametrize('string_char', ['S', 'U'])
def test_string_to_string_cancast(self, other_dt, string_char):
    other_dt = np.dtype(other_dt)
    fact = 1 if string_char == 'S' else 4
    div = 1 if other_dt.char == 'S' else 4
    string_DT = type(np.dtype(string_char))
    cast = get_castingimpl(type(other_dt), string_DT)
    expected_length = other_dt.itemsize // div
    string_dt = np.dtype(f'{string_char}{expected_length}')
    safety, (res_other_dt, res_dt), view_off = cast._resolve_descriptors((other_dt, None))
    assert res_dt.itemsize == expected_length * fact
    assert isinstance(res_dt, string_DT)
    expected_view_off = None
    if other_dt.char == string_char:
        if other_dt.isnative:
            expected_safety = Casting.no
            expected_view_off = 0
        else:
            expected_safety = Casting.equiv
    elif string_char == 'U':
        expected_safety = Casting.safe
    else:
        expected_safety = Casting.unsafe
    assert view_off == expected_view_off
    assert expected_safety == safety
    for change_length in [-1, 0, 1]:
        to_dt = self.string_with_modified_length(string_dt, change_length)
        safety, (_, res_dt), view_off = cast._resolve_descriptors((other_dt, to_dt))
        assert res_dt is to_dt
        if change_length <= 0:
            assert view_off == expected_view_off
        else:
            assert view_off is None
        if expected_safety == Casting.unsafe:
            assert safety == expected_safety
        elif change_length < 0:
            assert safety == Casting.same_kind
        elif change_length == 0:
            assert safety == expected_safety
        elif change_length > 0:
            assert safety == Casting.safe