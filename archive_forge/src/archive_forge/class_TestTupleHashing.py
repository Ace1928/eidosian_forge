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
class TestTupleHashing(BaseTest):
    """
    Test hashing of tuples.
    """

    def check_tuples(self, value_generator, split):
        for values in value_generator:
            tuples = [split(a) for a in values]
            self.check_hash_values(tuples)

    def test_homogeneous_tuples(self):
        typ = np.uint64

        def split2(i):
            """
            Split i's bits into 2 integers.
            """
            i = typ(i)
            return (i & typ(6148914691236517205), i & typ(12297829382473034410))

        def split3(i):
            """
            Split i's bits into 3 integers.
            """
            i = typ(i)
            return (i & typ(2635249153387078802), i & typ(5270498306774157604), i & typ(10540996613548315209))
        self.check_tuples(self.int_samples(), split2)
        self.check_tuples(self.int_samples(), split3)
        self.check_hash_values([(7,), (0,), (0, 0), (0.5,), (0.5, (7,), (-2, 3, (4, 6)))])

    def test_heterogeneous_tuples(self):
        modulo = 2 ** 63

        def split(i):
            a = i & 6148914691236517205
            b = i & 2863311530 ^ i >> 32 & 2863311530
            return (np.int64(a), np.float64(b * 0.0001))
        self.check_tuples(self.int_samples(), split)