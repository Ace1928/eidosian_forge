import os
import platform
import re
import subprocess
import sys
import threading
from itertools import permutations
from numba import njit, gdb, gdb_init, gdb_breakpoint, prange
from numba.core import errors
from numba import jit
from numba.tests.support import (TestCase, captured_stdout, tag,
from numba.tests.gdb_support import needs_gdb
import unittest
@not_arm
@unix_only
class TestGdbBindImpls(TestCase):
    """
    Contains unit test implementations for gdb binding testing. Test must be
    decorated with `@needs_gdb_harness` to prevent their running under normal
    test conditions, the test methods must also end with `_impl` to be
    considered for execution. The tests themselves are invoked by the
    `TestGdbBinding` test class through the parsing of this class for test
    methods and then running the discovered tests in a separate process. Test
    names not including the word `quick` will be tagged as @tag('long_running')
    """

    @needs_gdb_harness
    def test_gdb_cmd_lang_cpython_quick_impl(self):
        with captured_stdout():
            impl_gdb_call(10)

    @needs_gdb_harness
    def test_gdb_cmd_lang_nopython_quick_impl(self):
        with captured_stdout():
            _dbg_njit(impl_gdb_call)(10)

    @needs_gdb_harness
    def test_gdb_cmd_lang_objmode_quick_impl(self):
        with captured_stdout():
            _dbg_jit(impl_gdb_call)(10)

    @needs_gdb_harness
    def test_gdb_split_init_and_break_cpython_impl(self):
        with captured_stdout():
            impl_gdb_call_w_bp(10)

    @needs_gdb_harness
    def test_gdb_split_init_and_break_nopython_impl(self):
        with captured_stdout():
            _dbg_njit(impl_gdb_call_w_bp)(10)

    @needs_gdb_harness
    def test_gdb_split_init_and_break_objmode_impl(self):
        with captured_stdout():
            _dbg_jit(impl_gdb_call_w_bp)(10)

    @skip_parfors_unsupported
    @needs_gdb_harness
    def test_gdb_split_init_and_break_w_parallel_cpython_impl(self):
        with captured_stdout():
            impl_gdb_split_init_and_break_w_parallel(10)

    @skip_parfors_unsupported
    @needs_gdb_harness
    def test_gdb_split_init_and_break_w_parallel_nopython_impl(self):
        with captured_stdout():
            _dbg_njit(impl_gdb_split_init_and_break_w_parallel)(10)

    @skip_parfors_unsupported
    @needs_gdb_harness
    def test_gdb_split_init_and_break_w_parallel_objmode_impl(self):
        with captured_stdout():
            _dbg_jit(impl_gdb_split_init_and_break_w_parallel)(10)