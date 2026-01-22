import os
import platform
import re
import textwrap
import warnings
import numpy as np
from numba.tests.support import (TestCase, override_config, override_env_config,
from numba import jit, njit
from numba.core import types, compiler, utils
from numba.core.errors import NumbaPerformanceWarning
from numba import prange
from numba.experimental import jitclass
import unittest
class TestGeneratorDebugOutput(DebugTestBase):
    func_name = 'simple_gen'

    def compile_simple_gen(self):
        with captured_stdout() as out:
            cfunc = njit((types.int64, types.int64))(simple_gen)
            self.assertPreciseEqual(list(cfunc(2, 5)), [2, 5])
        return out.getvalue()

    def test_dump_ir_generator(self):
        with override_config('DUMP_IR', True):
            out = self.compile_simple_gen()
        self.check_debug_output(out, ['ir'])
        self.assertIn('--GENERATOR INFO: %s' % self.func_name, out)
        expected_gen_info = textwrap.dedent("\n            generator state variables: ['x', 'y']\n            yield point #1: live variables = ['y'], weak live variables = ['x']\n            yield point #2: live variables = [], weak live variables = ['y']\n            ")
        self.assertIn(expected_gen_info, out)