import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def _int_clip(self, seq):
    if self._is_fp():
        return seq
    max_int = self._int_max()
    min_int = self._int_min()
    return [min(max(v, min_int), max_int) for v in seq]