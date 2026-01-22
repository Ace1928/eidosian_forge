import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
class _SIMD_INT(_Test_Utility):
    """
    To test all integer vector types at once
    """

    def test_operators_shift(self):
        if self.sfx in ('u8', 's8'):
            return
        data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
        for count in range(self._scalar_size()):
            data_shl_a = self.load([a << count for a in data_a])
            shl = self.shl(vdata_a, count)
            assert shl == data_shl_a
            data_shr_a = self.load([a >> count for a in data_a])
            shr = self.shr(vdata_a, count)
            assert shr == data_shr_a
        for count in range(1, self._scalar_size()):
            data_shl_a = self.load([a << count for a in data_a])
            shli = self.shli(vdata_a, count)
            assert shli == data_shl_a
            data_shr_a = self.load([a >> count for a in data_a])
            shri = self.shri(vdata_a, count)
            assert shri == data_shr_a

    def test_arithmetic_subadd_saturated(self):
        if self.sfx in ('u32', 's32', 'u64', 's64'):
            return
        data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
        data_adds = self._int_clip([a + b for a, b in zip(data_a, data_b)])
        adds = self.adds(vdata_a, vdata_b)
        assert adds == data_adds
        data_subs = self._int_clip([a - b for a, b in zip(data_a, data_b)])
        subs = self.subs(vdata_a, vdata_b)
        assert subs == data_subs

    def test_math_max_min(self):
        data_a = self._data()
        data_b = self._data(self.nlanes)
        vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
        data_max = [max(a, b) for a, b in zip(data_a, data_b)]
        simd_max = self.max(vdata_a, vdata_b)
        assert simd_max == data_max
        data_min = [min(a, b) for a, b in zip(data_a, data_b)]
        simd_min = self.min(vdata_a, vdata_b)
        assert simd_min == data_min

    @pytest.mark.parametrize('start', [-100, -10000, 0, 100, 10000])
    def test_reduce_max_min(self, start):
        """
        Test intrinsics:
            npyv_reduce_max_##sfx
            npyv_reduce_min_##sfx
        """
        vdata_a = self.load(self._data(start))
        assert self.reduce_max(vdata_a) == max(vdata_a)
        assert self.reduce_min(vdata_a) == min(vdata_a)