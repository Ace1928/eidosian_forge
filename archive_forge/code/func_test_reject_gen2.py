from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_reject_gen2(self):
    self.check_no_lift_generator(reject_gen2, (types.intp,), (123,))