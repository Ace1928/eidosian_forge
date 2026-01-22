import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def _test_type_repr(self, t):
    finfo = np.finfo(t)
    last_fraction_bit_idx = finfo.nexp + finfo.nmant
    last_exponent_bit_idx = finfo.nexp
    storage_bytes = np.dtype(t).itemsize * 8
    for which in ['small denorm', 'small norm']:
        constr = np.array([0] * storage_bytes, dtype=np.uint8)
        if which == 'small denorm':
            byte = last_fraction_bit_idx // 8
            bytebit = 7 - last_fraction_bit_idx % 8
            constr[byte] = 1 << bytebit
        elif which == 'small norm':
            byte = last_exponent_bit_idx // 8
            bytebit = 7 - last_exponent_bit_idx % 8
            constr[byte] = 1 << bytebit
        else:
            raise ValueError('hmm')
        val = constr.view(t)[0]
        val_repr = repr(val)
        val2 = t(eval(val_repr))
        if not (val2 == 0 and val < 1e-100):
            assert_equal(val, val2)