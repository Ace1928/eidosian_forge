import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('string_char', ['S', 'U'])
@pytest.mark.parametrize('other_dt', simple_dtype_instances())
def test_simple_string_casts_roundtrip(self, other_dt, string_char):
    """
        Tests casts from and to string by checking the roundtripping property.

        The test also covers some string to string casts (but not all).

        If this test creates issues, it should possibly just be simplified
        or even removed (checking whether unaligned/non-contiguous casts give
        the same results is useful, though).
        """
    string_DT = type(np.dtype(string_char))
    cast = get_castingimpl(type(other_dt), string_DT)
    cast_back = get_castingimpl(string_DT, type(other_dt))
    _, (res_other_dt, string_dt), _ = cast._resolve_descriptors((other_dt, None))
    if res_other_dt is not other_dt:
        assert other_dt.byteorder != res_other_dt.byteorder
        return
    orig_arr, values = self.get_data(other_dt, None)
    str_arr = np.zeros(len(orig_arr), dtype=string_dt)
    string_dt_short = self.string_with_modified_length(string_dt, -1)
    str_arr_short = np.zeros(len(orig_arr), dtype=string_dt_short)
    string_dt_long = self.string_with_modified_length(string_dt, 1)
    str_arr_long = np.zeros(len(orig_arr), dtype=string_dt_long)
    assert not cast._supports_unaligned
    assert not cast_back._supports_unaligned
    for contig in [True, False]:
        other_arr, str_arr = self.get_data_variation(orig_arr, str_arr, True, contig)
        _, str_arr_short = self.get_data_variation(orig_arr, str_arr_short.copy(), True, contig)
        _, str_arr_long = self.get_data_variation(orig_arr, str_arr_long, True, contig)
        cast._simple_strided_call((other_arr, str_arr))
        cast._simple_strided_call((other_arr, str_arr_short))
        assert_array_equal(str_arr.astype(string_dt_short), str_arr_short)
        cast._simple_strided_call((other_arr, str_arr_long))
        assert_array_equal(str_arr, str_arr_long)
        if other_dt.kind == 'b':
            continue
        other_arr[...] = 0
        cast_back._simple_strided_call((str_arr, other_arr))
        assert_array_equal(orig_arr, other_arr)
        other_arr[...] = 0
        cast_back._simple_strided_call((str_arr_long, other_arr))
        assert_array_equal(orig_arr, other_arr)