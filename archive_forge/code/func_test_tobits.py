import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_tobits(self):
    data2bits = lambda data: sum([int(x != 0) << i for i, x in enumerate(data, 0)])
    for data in (self._data(), self._data(reverse=True)):
        vdata = self._load_b(data)
        data_bits = data2bits(data)
        tobits = self.tobits(vdata)
        bin_tobits = bin(tobits)
        assert bin_tobits == bin(data_bits)