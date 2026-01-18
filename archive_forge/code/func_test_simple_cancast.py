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
def test_simple_cancast(self, from_Dt):
    for to_Dt in simple_dtypes:
        cast = get_castingimpl(from_Dt, to_Dt)
        for from_dt in [from_Dt(), from_Dt().newbyteorder()]:
            default = cast._resolve_descriptors((from_dt, None))[1][1]
            assert default == to_Dt()
            del default
            for to_dt in [to_Dt(), to_Dt().newbyteorder()]:
                casting, (from_res, to_res), view_off = cast._resolve_descriptors((from_dt, to_dt))
                assert type(from_res) == from_Dt
                assert type(to_res) == to_Dt
                if view_off is not None:
                    assert casting == Casting.no
                    assert Casting.equiv == CAST_TABLE[from_Dt][to_Dt]
                    assert from_res.isnative == to_res.isnative
                else:
                    if from_Dt == to_Dt:
                        assert from_res.isnative != to_res.isnative
                    assert casting == CAST_TABLE[from_Dt][to_Dt]
                if from_Dt is to_Dt:
                    assert from_dt is from_res
                    assert to_dt is to_res