import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize('dtype', np.typecodes['Integer'])
def test_signed_division_overflow(self, dtype):
    to_check = interesting_binop_operands(np.iinfo(dtype).min, -1, dtype)
    for op1, op2, extractor, operand_identifier in to_check:
        with pytest.warns(RuntimeWarning, match='overflow encountered'):
            res = op1 // op2
        assert res.dtype == op1.dtype
        assert extractor(res) == np.iinfo(op1.dtype).min
        res = op1 % op2
        assert res.dtype == op1.dtype
        assert extractor(res) == 0
        res = np.fmod(op1, op2)
        assert extractor(res) == 0
        with pytest.warns(RuntimeWarning, match='overflow encountered'):
            res1, res2 = np.divmod(op1, op2)
        assert res1.dtype == res2.dtype == op1.dtype
        assert extractor(res1) == np.iinfo(op1.dtype).min
        assert extractor(res2) == 0