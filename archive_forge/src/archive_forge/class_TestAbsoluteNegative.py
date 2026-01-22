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
class TestAbsoluteNegative:

    def test_abs_neg_blocked(self):
        for dt, sz in [(np.float32, 11), (np.float64, 5)]:
            for out, inp, msg in _gen_alignment_data(dtype=dt, type='unary', max_size=sz):
                tgt = [ncu.absolute(i) for i in inp]
                np.absolute(inp, out=out)
                assert_equal(out, tgt, err_msg=msg)
                assert_((out >= 0).all())
                tgt = [-1 * i for i in inp]
                np.negative(inp, out=out)
                assert_equal(out, tgt, err_msg=msg)
                for v in [np.nan, -np.inf, np.inf]:
                    for i in range(inp.size):
                        d = np.arange(inp.size, dtype=dt)
                        inp[:] = -d
                        inp[i] = v
                        d[i] = -v if v == -np.inf else v
                        assert_array_equal(np.abs(inp), d, err_msg=msg)
                        np.abs(inp, out=out)
                        assert_array_equal(out, d, err_msg=msg)
                        assert_array_equal(-inp, -1 * inp, err_msg=msg)
                        d = -1 * inp
                        np.negative(inp, out=out)
                        assert_array_equal(out, d, err_msg=msg)

    def test_lower_align(self):
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        assert_equal(np.abs(d), d)
        assert_equal(np.negative(d), -d)
        np.negative(d, out=d)
        np.negative(np.ones_like(d), out=d)
        np.abs(d, out=d)
        np.abs(np.ones_like(d), out=d)

    @pytest.mark.parametrize('dtype', ['d', 'f', 'int32', 'int64'])
    @pytest.mark.parametrize('big', [True, False])
    def test_noncontiguous(self, dtype, big):
        data = np.array([-1.0, 1.0, -0.0, 0.0, 2.2251e-308, -2.5, 2.5, -6, 6, -2.2251e-308, -8, 10], dtype=dtype)
        expect = np.array([1.0, -1.0, 0.0, -0.0, -2.2251e-308, 2.5, -2.5, 6, -6, 2.2251e-308, 8, -10], dtype=dtype)
        if big:
            data = np.repeat(data, 10)
            expect = np.repeat(expect, 10)
        out = np.ndarray(data.shape, dtype=dtype)
        ncontig_in = data[1::2]
        ncontig_out = out[1::2]
        contig_in = np.array(ncontig_in)
        assert_array_equal(np.negative(contig_in), expect[1::2])
        assert_array_equal(np.negative(contig_in, out=ncontig_out), expect[1::2])
        assert_array_equal(np.negative(ncontig_in), expect[1::2])
        assert_array_equal(np.negative(ncontig_in, out=ncontig_out), expect[1::2])
        data_split = np.array(np.array_split(data, 2))
        expect_split = np.array(np.array_split(expect, 2))
        assert_equal(np.negative(data_split), expect_split)