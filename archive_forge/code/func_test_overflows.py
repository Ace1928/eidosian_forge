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
@pytest.mark.parametrize('dividend_dtype', np.sctypes['int'])
@pytest.mark.parametrize('divisor_dtype', np.sctypes['int'])
@pytest.mark.parametrize('operation', [np.remainder, np.fmod, np.divmod, np.floor_divide, operator.mod, operator.floordiv])
@np.errstate(divide='warn', over='warn')
def test_overflows(self, dividend_dtype, divisor_dtype, operation):
    arrays = [np.array([np.iinfo(dividend_dtype).min] * i, dtype=dividend_dtype) for i in range(1, 129)]
    divisor = np.array([-1], dtype=divisor_dtype)
    if np.dtype(dividend_dtype).itemsize >= np.dtype(divisor_dtype).itemsize and operation in (np.divmod, np.floor_divide, operator.floordiv):
        with pytest.warns(RuntimeWarning, match='overflow encountered in'):
            result = operation(dividend_dtype(np.iinfo(dividend_dtype).min), divisor_dtype(-1))
            assert result == self.overflow_results[operation].nocast(dividend_dtype)
        for a in arrays:
            with pytest.warns(RuntimeWarning, match='overflow encountered in'):
                result = np.array(operation(a, divisor)).flatten('f')
                expected_array = np.array([self.overflow_results[operation].nocast(dividend_dtype)] * len(a)).flatten()
                assert_array_equal(result, expected_array)
    else:
        result = operation(dividend_dtype(np.iinfo(dividend_dtype).min), divisor_dtype(-1))
        assert result == self.overflow_results[operation].casted(dividend_dtype)
        for a in arrays:
            result = np.array(operation(a, divisor)).flatten('f')
            expected_array = np.array([self.overflow_results[operation].casted(dividend_dtype)] * len(a)).flatten()
            assert_array_equal(result, expected_array)