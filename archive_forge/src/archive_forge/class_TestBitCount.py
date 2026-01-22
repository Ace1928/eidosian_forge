import fractions
import platform
import types
from typing import Any, Type
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, IS_MUSL
class TestBitCount:

    @pytest.mark.parametrize('itype', np.sctypes['int'] + np.sctypes['uint'])
    def test_small(self, itype):
        for a in range(max(np.iinfo(itype).min, 0), 128):
            msg = f'Smoke test for {itype}({a}).bit_count()'
            assert itype(a).bit_count() == bin(a).count('1'), msg

    def test_bit_count(self):
        for exp in [10, 17, 63]:
            a = 2 ** exp
            assert np.uint64(a).bit_count() == 1
            assert np.uint64(a - 1).bit_count() == exp
            assert np.uint64(a ^ 63).bit_count() == 7
            assert np.uint64(a - 1 ^ 510).bit_count() == exp - 8