from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft
class TestNextFastLen:

    def test_next_fast_len(self):
        np.random.seed(1234)

        def nums():
            yield from range(1, 1000)
            yield (2 ** 5 * 3 ** 5 * 4 ** 5 + 1)
        for n in nums():
            m = next_fast_len(n)
            _assert_n_smooth(m, 11)
            assert m == next_fast_len(n, False)
            m = next_fast_len(n, True)
            _assert_n_smooth(m, 5)

    def test_np_integers(self):
        ITYPES = [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]
        for ityp in ITYPES:
            x = ityp(12345)
            testN = next_fast_len(x)
            assert_equal(testN, next_fast_len(int(x)))

    def testnext_fast_len_small(self):
        hams = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8, 8: 8, 14: 15, 15: 15, 16: 16, 17: 18, 1021: 1024, 1536: 1536, 51200000: 51200000}
        for x, y in hams.items():
            assert_equal(next_fast_len(x, True), y)

    @pytest.mark.xfail(sys.maxsize < 2 ** 32, reason='Hamming Numbers too large for 32-bit', raises=ValueError, strict=True)
    def testnext_fast_len_big(self):
        hams = {510183360: 510183360, 510183360 + 1: 512000000, 511000000: 512000000, 854296875: 854296875, 854296875 + 1: 859963392, 196608000000: 196608000000, 196608000000 + 1: 196830000000, 8789062500000: 8789062500000, 8789062500000 + 1: 8796093022208, 206391214080000: 206391214080000, 206391214080000 + 1: 206624260800000, 470184984576000: 470184984576000, 470184984576000 + 1: 470715894135000, 7222041363087360: 7222041363087360, 7222041363087360 + 1: 7230196133913600, 11920928955078125: 11920928955078125, 11920928955078125 - 1: 11920928955078125, 16677181699666569: 16677181699666569, 16677181699666569 - 1: 16677181699666569, 18014398509481984: 18014398509481984, 18014398509481984 - 1: 18014398509481984, 19200000000000000: 19200000000000000, 19200000000000000 + 1: 19221679687500000, 288230376151711744: 288230376151711744, 288230376151711744 + 1: 288325195312500000, 288325195312500000 - 1: 288325195312500000, 288325195312500000: 288325195312500000, 288325195312500000 + 1: 288555831593533440}
        for x, y in hams.items():
            assert_equal(next_fast_len(x, True), y)

    def test_keyword_args(self):
        assert next_fast_len(11, real=True) == 12
        assert next_fast_len(target=7, real=False) == 7