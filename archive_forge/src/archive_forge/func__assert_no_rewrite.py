import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def _assert_no_rewrite(self, control_ir, test_ir):
    """
        Given two dictionaries of Numba IR blocks, check to make sure
        the control IR and the test IR both have no array expressions.
        """
    self.assertEqual(len(control_ir), len(test_ir))
    for k, v in control_ir.items():
        control_block = v.body
        test_block = test_ir[k].body
        self.assertEqual(len(control_block), len(test_block))
        self._assert_array_exprs(control_block, 0)
        self._assert_array_exprs(test_block, 0)