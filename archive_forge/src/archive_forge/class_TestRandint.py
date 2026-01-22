import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
class TestRandint:
    rfunc = random.randint
    itype = [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]

    def test_unsupported_type(self):
        assert_raises(TypeError, self.rfunc, 1, dtype=float)

    def test_bounds_checking(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, dtype=dt)

    def test_rng_zero_and_extremes(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)
            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)
            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

    def test_full_range(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            try:
                self.rfunc(lbnd, ubnd, dtype=dt)
            except Exception as e:
                raise AssertionError('No error should have been raised, but one was with the following message:\n\n%s' % str(e))

    def test_in_bounds_fuzz(self):
        random.seed()
        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd, size=2 ** 16, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)
        vals = self.rfunc(0, 2, size=2 ** 16, dtype=np.bool_)
        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def test_repeatability(self):
        tgt = {'bool': '509aea74d792fb931784c4b0135392c65aec64beee12b0cc167548a2c3d31e71', 'int16': '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4', 'int32': 'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f', 'int64': '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e', 'int8': '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404', 'uint16': '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4', 'uint32': 'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f', 'uint64': '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e', 'uint8': '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404'}
        for dt in self.itype[1:]:
            random.seed(1234)
            if sys.byteorder == 'little':
                val = self.rfunc(0, 6, size=1000, dtype=dt)
            else:
                val = self.rfunc(0, 6, size=1000, dtype=dt).byteswap()
            res = hashlib.sha256(val.view(np.int8)).hexdigest()
            assert_(tgt[np.dtype(dt).name] == res)
        random.seed(1234)
        val = self.rfunc(0, 2, size=1000, dtype=bool).view(np.int8)
        res = hashlib.sha256(val).hexdigest()
        assert_(tgt[np.dtype(bool).name] == res)

    @pytest.mark.skipif(np.iinfo('l').max < 2 ** 32, reason='Cannot test with 32-bit C long')
    def test_repeatability_32bit_boundary_broadcasting(self):
        desired = np.array([[[3992670689, 2438360420, 2557845020], [4107320065, 4142558326, 3216529513], [1605979228, 2807061240, 665605495]], [[3211410639, 4128781000, 457175120], [1712592594, 1282922662, 3081439808], [3997822960, 2008322436, 1563495165]], [[1398375547, 4269260146, 115316740], [3414372578, 3437564012, 2112038651], [3572980305, 2260248732, 3908238631]], [[2561372503, 223155946, 3127879445], [441282060, 3514786552, 2148440361], [1629275283, 3479737011, 3003195987]], [[412181688, 940383289, 3047321305], [2978368172, 764731833, 2282559898], [105711276, 720447391, 3596512484]]])
        for size in [None, (5, 3, 3)]:
            random.seed(12345)
            x = self.rfunc([[-1], [0], [1]], [2 ** 32 - 1, 2 ** 32, 2 ** 32 + 1], size=size)
            assert_array_equal(x, desired if size is not None else desired[0])

    def test_int64_uint64_corner_case(self):
        dt = np.int64
        tgt = np.iinfo(np.int64).max
        lbnd = np.int64(np.iinfo(np.int64).max)
        ubnd = np.uint64(np.iinfo(np.int64).max + 1)
        actual = random.randint(lbnd, ubnd, dtype=dt)
        assert_equal(actual, tgt)

    def test_respect_dtype_singleton(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            assert_equal(sample.dtype, np.dtype(dt))
        for dt in (bool, int):
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            assert_(not hasattr(sample, 'dtype'))
            assert_equal(type(sample), dt)