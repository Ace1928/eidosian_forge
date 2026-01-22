import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def _load_b(self, data):
    len_str = self.sfx[1:]
    load = getattr(self.npyv, 'load_u' + len_str)
    cvt = getattr(self.npyv, f'cvt_b{len_str}_u{len_str}')
    return cvt(load(data))