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
@not_unix
class TestGdbExceptions(TestCase):

    def test_call_gdb(self):

        def nop_compiler(x):
            return x
        for compiler in [nop_compiler, jit(forceobj=True), njit]:
            for meth in [gdb, gdb_init]:

                def python_func():
                    meth()
                with self.assertRaises(errors.TypingError) as raises:
                    compiler(python_func)()
                msg = 'gdb support is only available on unix-like systems'
                self.assertIn(msg, str(raises.exception))