import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
class TestIntegers:
    rfunc = random.integers
    itype = [bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]

    def test_unsupported_type(self, endpoint):
        assert_raises(TypeError, self.rfunc, 1, endpoint=endpoint, dtype=float)

    def test_bounds_checking(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd - 1], ubnd, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd], [ubnd + 1], endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [ubnd], [lbnd], endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, [0], endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [ubnd + 1], [ubnd], endpoint=endpoint, dtype=dt)

    def test_bounds_checking_array(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + (not endpoint)
            assert_raises(ValueError, self.rfunc, [lbnd - 1] * 2, [ubnd] * 2, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd] * 2, [ubnd + 1] * 2, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, [lbnd] * 2, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [1] * 2, 0, endpoint=endpoint, dtype=dt)

    def test_rng_zero_and_extremes(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            is_open = not endpoint
            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000, endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], tgt + is_open, size=1000, endpoint=endpoint, dtype=dt), tgt)
            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000, endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc(tgt, [tgt + is_open], size=1000, endpoint=endpoint, dtype=dt), tgt)
            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000, endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], [tgt + is_open], size=1000, endpoint=endpoint, dtype=dt), tgt)

    def test_rng_zero_and_extremes_array(self, endpoint):
        size = 1000
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            tgt = ubnd - 1
            assert_equal(self.rfunc([tgt], [tgt + 1], size=size, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)
            tgt = lbnd
            assert_equal(self.rfunc([tgt], [tgt + 1], size=size, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)
            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc([tgt], [tgt + 1], size=size, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

    def test_full_range(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            try:
                self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            except Exception as e:
                raise AssertionError('No error should have been raised, but one was with the following message:\n\n%s' % str(e))

    def test_full_range_array(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            try:
                self.rfunc([lbnd] * 2, [ubnd], endpoint=endpoint, dtype=dt)
            except Exception as e:
                raise AssertionError('No error should have been raised, but one was with the following message:\n\n%s' % str(e))

    def test_in_bounds_fuzz(self, endpoint):
        random = Generator(MT19937())
        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd - endpoint, size=2 ** 16, endpoint=endpoint, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)
        vals = self.rfunc(0, 2 - endpoint, size=2 ** 16, endpoint=endpoint, dtype=bool)
        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def test_scalar_array_equiv(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            size = 1000
            random = Generator(MT19937(1234))
            scalar = random.integers(lbnd, ubnd, size=size, endpoint=endpoint, dtype=dt)
            random = Generator(MT19937(1234))
            scalar_array = random.integers([lbnd], [ubnd], size=size, endpoint=endpoint, dtype=dt)
            random = Generator(MT19937(1234))
            array = random.integers([lbnd] * size, [ubnd] * size, size=size, endpoint=endpoint, dtype=dt)
            assert_array_equal(scalar, scalar_array)
            assert_array_equal(scalar, array)

    def test_repeatability(self, endpoint):
        tgt = {'bool': '053594a9b82d656f967c54869bc6970aa0358cf94ad469c81478459c6a90eee3', 'int16': '54de9072b6ee9ff7f20b58329556a46a447a8a29d67db51201bf88baa6e4e5d4', 'int32': 'd3a0d5efb04542b25ac712e50d21f39ac30f312a5052e9bbb1ad3baa791ac84b', 'int64': '14e224389ac4580bfbdccb5697d6190b496f91227cf67df60989de3d546389b1', 'int8': '0e203226ff3fbbd1580f15da4621e5f7164d0d8d6b51696dd42d004ece2cbec1', 'uint16': '54de9072b6ee9ff7f20b58329556a46a447a8a29d67db51201bf88baa6e4e5d4', 'uint32': 'd3a0d5efb04542b25ac712e50d21f39ac30f312a5052e9bbb1ad3baa791ac84b', 'uint64': '14e224389ac4580bfbdccb5697d6190b496f91227cf67df60989de3d546389b1', 'uint8': '0e203226ff3fbbd1580f15da4621e5f7164d0d8d6b51696dd42d004ece2cbec1'}
        for dt in self.itype[1:]:
            random = Generator(MT19937(1234))
            if sys.byteorder == 'little':
                val = random.integers(0, 6 - endpoint, size=1000, endpoint=endpoint, dtype=dt)
            else:
                val = random.integers(0, 6 - endpoint, size=1000, endpoint=endpoint, dtype=dt).byteswap()
            res = hashlib.sha256(val).hexdigest()
            assert_(tgt[np.dtype(dt).name] == res)
        random = Generator(MT19937(1234))
        val = random.integers(0, 2 - endpoint, size=1000, endpoint=endpoint, dtype=bool).view(np.int8)
        res = hashlib.sha256(val).hexdigest()
        assert_(tgt[np.dtype(bool).name] == res)

    def test_repeatability_broadcasting(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt in (bool, np.bool_) else np.iinfo(dt).min
            ubnd = 2 if dt in (bool, np.bool_) else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            random = Generator(MT19937(1234))
            val = random.integers(lbnd, ubnd, size=1000, endpoint=endpoint, dtype=dt)
            random = Generator(MT19937(1234))
            val_bc = random.integers([lbnd] * 1000, ubnd, endpoint=endpoint, dtype=dt)
            assert_array_equal(val, val_bc)
            random = Generator(MT19937(1234))
            val_bc = random.integers([lbnd] * 1000, [ubnd] * 1000, endpoint=endpoint, dtype=dt)
            assert_array_equal(val, val_bc)

    @pytest.mark.parametrize('bound, expected', [(2 ** 32 - 1, np.array([517043486, 1364798665, 1733884389, 1353720612, 3769704066, 1170797179, 4108474671])), (2 ** 32, np.array([517043487, 1364798666, 1733884390, 1353720613, 3769704067, 1170797180, 4108474672])), (2 ** 32 + 1, np.array([517043487, 1733884390, 3769704068, 4108474673, 1831631863, 1215661561, 3869512430]))])
    def test_repeatability_32bit_boundary(self, bound, expected):
        for size in [None, len(expected)]:
            random = Generator(MT19937(1234))
            x = random.integers(bound, size=size)
            assert_equal(x, expected if size is not None else expected[0])

    def test_repeatability_32bit_boundary_broadcasting(self):
        desired = np.array([[[1622936284, 3620788691, 1659384060], [1417365545, 760222891, 1909653332], [3788118662, 660249498, 4092002593]], [[3625610153, 2979601262, 3844162757], [685800658, 120261497, 2694012896], [1207779440, 1586594375, 3854335050]], [[3004074748, 2310761796, 3012642217], [2067714190, 2786677879, 1363865881], [791663441, 1867303284, 2169727960]], [[1939603804, 1250951100, 298950036], [1040128489, 3791912209, 3317053765], [3155528714, 61360675, 2305155588]], [[817688762, 1335621943, 3288952434], [1770890872, 1102951817, 1957607470], [3099996017, 798043451, 48334215]]])
        for size in [None, (5, 3, 3)]:
            random = Generator(MT19937(12345))
            x = random.integers([[-1], [0], [1]], [2 ** 32 - 1, 2 ** 32, 2 ** 32 + 1], size=size)
            assert_array_equal(x, desired if size is not None else desired[0])

    def test_int64_uint64_broadcast_exceptions(self, endpoint):
        configs = {np.uint64: ((0, 2 ** 65), (-1, 2 ** 62), (10, 9), (0, 0)), np.int64: ((0, 2 ** 64), (-2 ** 64, 2 ** 62), (10, 9), (0, 0), (-2 ** 63 - 1, -2 ** 63 - 1))}
        for dtype in configs:
            for config in configs[dtype]:
                low, high = config
                high = high - endpoint
                low_a = np.array([[low] * 10])
                high_a = np.array([high] * 10)
                assert_raises(ValueError, random.integers, low, high, endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high, endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_a, endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high_a, endpoint=endpoint, dtype=dtype)
                low_o = np.array([[low] * 10], dtype=object)
                high_o = np.array([high] * 10, dtype=object)
                assert_raises(ValueError, random.integers, low_o, high, endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_o, endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_o, high_o, endpoint=endpoint, dtype=dtype)

    def test_int64_uint64_corner_case(self, endpoint):
        dt = np.int64
        tgt = np.iinfo(np.int64).max
        lbnd = np.int64(np.iinfo(np.int64).max)
        ubnd = np.uint64(np.iinfo(np.int64).max + 1 - endpoint)
        actual = random.integers(lbnd, ubnd, endpoint=endpoint, dtype=dt)
        assert_equal(actual, tgt)

    def test_respect_dtype_singleton(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            dt = np.bool_ if dt is bool else dt
            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)
        for dt in (bool, int):
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            assert not hasattr(sample, 'dtype')
            assert_equal(type(sample), dt)

    def test_respect_dtype_array(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            dt = np.bool_ if dt is bool else dt
            sample = self.rfunc([lbnd], [ubnd], endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)
            sample = self.rfunc([lbnd] * 2, [ubnd] * 2, endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)

    def test_zero_size(self, endpoint):
        for dt in self.itype:
            sample = self.rfunc(0, 0, (3, 0, 4), endpoint=endpoint, dtype=dt)
            assert sample.shape == (3, 0, 4)
            assert sample.dtype == dt
            assert self.rfunc(0, -10, 0, endpoint=endpoint, dtype=dt).shape == (0,)
            assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
            assert_equal(random.integers(0, -10, size=0).shape, (0,))
            assert_equal(random.integers(10, 10, size=0).shape, (0,))

    def test_error_byteorder(self):
        other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
        with pytest.raises(ValueError):
            random.integers(0, 200, size=10, dtype=other_byteord_dt)

    @pytest.mark.slow
    @pytest.mark.parametrize('sample_size,high,dtype,chi2max', [(5000000, 5, np.int8, 125.0), (5000000, 7, np.uint8, 150.0), (10000000, 2500, np.int16, 3300.0), (50000000, 5000, np.uint16, 6500.0)])
    def test_integers_small_dtype_chisquared(self, sample_size, high, dtype, chi2max):
        samples = random.integers(high, size=sample_size, dtype=dtype)
        values, counts = np.unique(samples, return_counts=True)
        expected = sample_size / high
        chi2 = ((counts - expected) ** 2 / expected).sum()
        assert chi2 < chi2max