import unittest
from unittest.case import TestCase
import warnings
import numpy as np
from numba import objmode
from numba.core import ir, compiler
from numba.core import errors
from numba.core.compiler import (
from numba.core.compiler_machinery import (
from numba.core.untyped_passes import (
from numba import njit
def test_staticraise(self):
    a = ir.StaticRaise(AssertionError, None, self.loc1)
    b = ir.StaticRaise(AssertionError, None, self.loc1)
    c = ir.StaticRaise(AssertionError, None, self.loc2)
    e = ir.StaticRaise(AssertionError, ('str',), self.loc1)
    d = ir.StaticRaise(RuntimeError, None, self.loc1)
    self.check(a, same=[b, c], different=[d, e])