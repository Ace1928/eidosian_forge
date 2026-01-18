import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('from_Dt', simple_dtypes)
def test_numeric_to_times(self, from_Dt):
    from_dt = from_Dt()
    time_dtypes = [np.dtype('M8'), np.dtype('M8[ms]'), np.dtype('M8[4D]'), np.dtype('m8'), np.dtype('m8[ms]'), np.dtype('m8[4D]')]
    for time_dt in time_dtypes:
        cast = get_castingimpl(type(from_dt), type(time_dt))
        casting, (from_res, to_res), view_off = cast._resolve_descriptors((from_dt, time_dt))
        assert from_res is from_dt
        assert to_res is time_dt
        del from_res, to_res
        assert casting & CAST_TABLE[from_Dt][type(time_dt)]
        assert view_off is None
        int64_dt = np.dtype(np.int64)
        arr1, arr2, values = self.get_data(from_dt, int64_dt)
        arr2 = arr2.view(time_dt)
        arr2[...] = np.datetime64('NaT')
        if time_dt == np.dtype('M8'):
            arr1[-1] = 0
            cast._simple_strided_call((arr1, arr2))
            with pytest.raises(ValueError):
                str(arr2[-1])
            return
        cast._simple_strided_call((arr1, arr2))
        assert [int(v) for v in arr2.tolist()] == values
        arr1_o, arr2_o = self.get_data_variation(arr1, arr2, True, False)
        cast._simple_strided_call((arr1_o, arr2_o))
        assert_array_equal(arr2_o, arr2)
        assert arr2_o.tobytes() == arr2.tobytes()