from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def try_lift(self, pyfunc, argtypes):
    from numba.core.registry import cpu_target
    cres = compile_extra(cpu_target.typing_context, cpu_target.target_context, pyfunc, argtypes, return_type=None, flags=looplift_flags, locals={})
    self.assertEqual(len(cres.lifted), 1)
    return cres