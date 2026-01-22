import unittest
import os
import sys
import subprocess
from collections import defaultdict
from textwrap import dedent
import numpy as np
from numba import jit, config, typed, typeof
from numba.core import types, utils
import unittest
from numba.tests.support import (TestCase, skip_unless_py10_or_later,
from numba.cpython.unicode import compile_time_get_string_data
from numba.cpython import hashing
class TestNumberHashing(BaseTest):
    """
    Test hashing of number types.
    """

    def check_floats(self, typ):
        for a in self.float_samples(typ):
            self.assertEqual(a.dtype, np.dtype(typ))
            self.check_hash_values(a)

    def check_complex(self, typ, float_ty):
        for a in self.complex_samples(typ, float_ty):
            self.assertEqual(a.dtype, np.dtype(typ))
            self.check_hash_values(a)

    def test_floats(self):
        self.check_floats(np.float32)
        self.check_floats(np.float64)

    def test_complex(self):
        self.check_complex(np.complex64, np.float32)
        self.check_complex(np.complex128, np.float64)

    def test_bool(self):
        self.check_hash_values([False, True])

    def test_ints(self):
        minmax = []
        for ty in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            for a in self.int_samples(ty):
                self.check_hash_values(a)
            info = np.iinfo(ty)
            self.check_hash_values([ty(-1)])
            self.check_hash_values([ty(0)])
            signed = 'uint' not in str(ty)
            sz = ty().itemsize
            for x in [info.min, info.max]:
                shifts = 8 * sz
                y = x
                for i in range(shifts):
                    twiddle1 = 12297829382473034410
                    twiddle2 = 6148914691236517205
                    vals = [y]
                    for tw in [twiddle1, twiddle2]:
                        val = y & twiddle1
                        if val < sys.maxsize:
                            vals.append(val)
                    for v in vals:
                        self.check_hash_values([ty(v)])
                    if signed:
                        for v in vals:
                            if v != info.min:
                                self.check_hash_values([ty(-v)])
                    if x == 0:
                        y = (y | 1) << 1
                    else:
                        y = y >> 1
        self.check_hash_values([np.int64(2305843009213693950)])
        self.check_hash_values([np.int64(2305843009213693951)])
        self.check_hash_values([np.uint64(2305843009213693950)])
        self.check_hash_values([np.uint64(2305843009213693951)])
        self.check_hash_values([np.int64(-9223372036854775807)])
        self.check_hash_values([np.int64(-9223372036854775798)])
        self.check_hash_values([np.int64(-9223372036854775708)])
        self.check_hash_values([np.int32(-2147483647)])
        self.check_hash_values([np.int32(-2147483638)])
        self.check_hash_values([np.int32(-2147483548)])

    @skip_unless_py10_or_later
    def test_py310_nan_hash(self):
        x = [float('nan') for i in range(10)]
        out = set([self.cfunc(z) for z in x])
        self.assertGreater(len(out), 1)