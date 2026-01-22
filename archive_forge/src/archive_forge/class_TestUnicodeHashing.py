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
class TestUnicodeHashing(BaseTest):

    def test_basic_unicode(self):
        kind1_string = 'abcdefghijklmnopqrstuvwxyz'
        for i in range(len(kind1_string)):
            self.check_hash_values([kind1_string[:i]])
        sep = '眼'
        kind2_string = sep.join(list(kind1_string))
        for i in range(len(kind2_string)):
            self.check_hash_values([kind2_string[:i]])
        sep = '🐍⚡'
        kind4_string = sep.join(list(kind1_string))
        for i in range(len(kind4_string)):
            self.check_hash_values([kind4_string[:i]])
        empty_string = ''
        self.check_hash_values(empty_string)

    def test_hash_passthrough(self):
        kind1_string = 'abcdefghijklmnopqrstuvwxyz'

        @jit(nopython=True)
        def fn(x):
            return x._hash
        hash_value = compile_time_get_string_data(kind1_string)[-1]
        self.assertTrue(hash_value != -1)
        self.assertEqual(fn(kind1_string), hash_value)

    def test_hash_passthrough_call(self):
        kind1_string = 'abcdefghijklmnopqrstuvwxyz'

        @jit(nopython=True)
        def fn(x):
            return (x._hash, hash(x))
        hash_value = compile_time_get_string_data(kind1_string)[-1]
        self.assertTrue(hash_value != -1)
        self.assertEqual(fn(kind1_string), (hash_value, hash_value))

    @unittest.skip('Needs hash computation at const unpickling time')
    def test_hash_literal(self):

        @jit(nopython=True)
        def fn():
            x = 'abcdefghijklmnopqrstuvwxyz'
            return x
        val = fn()
        tmp = hash('abcdefghijklmnopqrstuvwxyz')
        self.assertEqual(tmp, compile_time_get_string_data(val)[-1])

    def test_hash_on_str_creation(self):

        def impl(do_hash):
            const1 = 'aaaa'
            const2 = '眼眼眼眼'
            new = const1 + const2
            if do_hash:
                hash(new)
            return new
        jitted = jit(nopython=True)(impl)
        compute_hash = False
        expected = impl(compute_hash)
        got = jitted(compute_hash)
        a = compile_time_get_string_data(expected)
        b = compile_time_get_string_data(got)
        self.assertEqual(a[:-1], b[:-1])
        self.assertTrue(a[-1] != b[-1])
        compute_hash = True
        expected = impl(compute_hash)
        got = jitted(compute_hash)
        a = compile_time_get_string_data(expected)
        b = compile_time_get_string_data(got)
        self.assertEqual(a, b)