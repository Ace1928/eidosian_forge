import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_sum_complex(self):
    for dt in (np.complex64, np.complex128, np.clongdouble):
        for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127, 128, 1024, 1235):
            tgt = dt(v * (v + 1) / 2) - dt(v * (v + 1) / 2 * 1j)
            d = np.empty(v, dtype=dt)
            d.real = np.arange(1, v + 1)
            d.imag = -np.arange(1, v + 1)
            assert_almost_equal(np.sum(d), tgt)
            assert_almost_equal(np.sum(d[::-1]), tgt)
        d = np.ones(500, dtype=dt) + 1j
        assert_almost_equal(np.sum(d[::2]), 250.0 + 250j)
        assert_almost_equal(np.sum(d[1::2]), 250.0 + 250j)
        assert_almost_equal(np.sum(d[::3]), 167.0 + 167j)
        assert_almost_equal(np.sum(d[1::3]), 167.0 + 167j)
        assert_almost_equal(np.sum(d[::-2]), 250.0 + 250j)
        assert_almost_equal(np.sum(d[-1::-2]), 250.0 + 250j)
        assert_almost_equal(np.sum(d[::-3]), 167.0 + 167j)
        assert_almost_equal(np.sum(d[-1::-3]), 167.0 + 167j)
        d = np.ones((1,), dtype=dt) + 1j
        d += d
        assert_almost_equal(d, 2.0 + 2j)