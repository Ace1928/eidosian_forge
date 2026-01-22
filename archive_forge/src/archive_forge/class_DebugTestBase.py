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
class DebugTestBase(TestCase):
    all_dumps = set(['bytecode', 'cfg', 'ir', 'typeinfer', 'llvm', 'func_opt_llvm', 'optimized_llvm', 'assembly'])

    def assert_fails(self, *args, **kwargs):
        self.assertRaises(AssertionError, *args, **kwargs)

    def check_debug_output(self, out, dump_names):
        enabled_dumps = dict.fromkeys(self.all_dumps, False)
        for name in dump_names:
            assert name in enabled_dumps
            enabled_dumps[name] = True
        for name, enabled in sorted(enabled_dumps.items()):
            check_meth = getattr(self, '_check_dump_%s' % name)
            if enabled:
                check_meth(out)
            else:
                self.assert_fails(check_meth, out)

    def _check_dump_bytecode(self, out):
        if utils.PYVERSION in ((3, 11), (3, 12)):
            self.assertIn('BINARY_OP', out)
        elif utils.PYVERSION in ((3, 9), (3, 10)):
            self.assertIn('BINARY_ADD', out)
        else:
            raise NotImplementedError(utils.PYVERSION)

    def _check_dump_cfg(self, out):
        self.assertIn('CFG dominators', out)

    def _check_dump_ir(self, out):
        self.assertIn('--IR DUMP: %s--' % self.func_name, out)

    def _check_dump_typeinfer(self, out):
        self.assertIn('--propagate--', out)

    def _check_dump_llvm(self, out):
        self.assertIn('--LLVM DUMP', out)
        if compiler.Flags.options['auto_parallel'].default.enabled == False:
            self.assertRegex(out, 'store i64 %\\"\\.\\d", i64\\* %"retptr"', out)

    def _check_dump_func_opt_llvm(self, out):
        self.assertIn('--FUNCTION OPTIMIZED DUMP %s' % self.func_name, out)
        self.assertIn('add nsw i64 %arg.somearg, 1', out)

    def _check_dump_optimized_llvm(self, out):
        self.assertIn('--OPTIMIZED DUMP %s' % self.func_name, out)
        self.assertIn('add nsw i64 %arg.somearg, 1', out)

    def _check_dump_assembly(self, out):
        self.assertIn('--ASSEMBLY %s' % self.func_name, out)
        if platform.machine() in ('x86_64', 'AMD64', 'i386', 'i686'):
            self.assertIn('xorl', out)