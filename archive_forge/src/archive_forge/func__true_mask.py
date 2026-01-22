import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def _true_mask(self):
    max_unsig = getattr(self.npyv, 'setall_u' + self.sfx[1:])(-1)
    return max_unsig[0]