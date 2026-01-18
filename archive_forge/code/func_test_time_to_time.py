import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize(['from_dt', 'to_dt', 'expected_casting', 'expected_view_off', 'nom', 'denom'], [('M8[ns]', None, Casting.no, 0, 1, 1), (str(np.dtype('M8[ns]').newbyteorder()), None, Casting.equiv, None, 1, 1), ('M8', 'M8[ms]', Casting.safe, 0, 1, 1), ('M8[ms]', 'M8', Casting.unsafe, None, 1, 1), ('M8[5ms]', 'M8[5ms]', Casting.no, 0, 1, 1), ('M8[ns]', 'M8[ms]', Casting.same_kind, None, 1, 10 ** 6), ('M8[ms]', 'M8[ns]', Casting.safe, None, 10 ** 6, 1), ('M8[ms]', 'M8[7ms]', Casting.same_kind, None, 1, 7), ('M8[4D]', 'M8[1M]', Casting.same_kind, None, None, [-2 ** 63, 0, -1, 1314, -1315, 564442610]), ('m8[ns]', None, Casting.no, 0, 1, 1), (str(np.dtype('m8[ns]').newbyteorder()), None, Casting.equiv, None, 1, 1), ('m8', 'm8[ms]', Casting.safe, 0, 1, 1), ('m8[ms]', 'm8', Casting.unsafe, None, 1, 1), ('m8[5ms]', 'm8[5ms]', Casting.no, 0, 1, 1), ('m8[ns]', 'm8[ms]', Casting.same_kind, None, 1, 10 ** 6), ('m8[ms]', 'm8[ns]', Casting.safe, None, 10 ** 6, 1), ('m8[ms]', 'm8[7ms]', Casting.same_kind, None, 1, 7), ('m8[4D]', 'm8[1M]', Casting.unsafe, None, None, [-2 ** 63, 0, 0, 1314, -1315, 564442610])])
def test_time_to_time(self, from_dt, to_dt, expected_casting, expected_view_off, nom, denom):
    from_dt = np.dtype(from_dt)
    if to_dt is not None:
        to_dt = np.dtype(to_dt)
    values = np.array([-2 ** 63, 1, 2 ** 63 - 1, 10000, -10000, 2 ** 32])
    values = values.astype(np.dtype('int64').newbyteorder(from_dt.byteorder))
    assert values.dtype.byteorder == from_dt.byteorder
    assert np.isnat(values.view(from_dt)[0])
    DType = type(from_dt)
    cast = get_castingimpl(DType, DType)
    casting, (from_res, to_res), view_off = cast._resolve_descriptors((from_dt, to_dt))
    assert from_res is from_dt
    assert to_res is to_dt or to_dt is None
    assert casting == expected_casting
    assert view_off == expected_view_off
    if nom is not None:
        expected_out = (values * nom // denom).view(to_res)
        expected_out[0] = 'NaT'
    else:
        expected_out = np.empty_like(values)
        expected_out[...] = denom
        expected_out = expected_out.view(to_dt)
    orig_arr = values.view(from_dt)
    orig_out = np.empty_like(expected_out)
    if casting == Casting.unsafe and (to_dt == 'm8' or to_dt == 'M8'):
        with pytest.raises(ValueError):
            cast._simple_strided_call((orig_arr, orig_out))
        return
    for aligned in [True, True]:
        for contig in [True, True]:
            arr, out = self.get_data_variation(orig_arr, orig_out, aligned, contig)
            out[...] = 0
            cast._simple_strided_call((arr, out))
            assert_array_equal(out.view('int64'), expected_out.view('int64'))