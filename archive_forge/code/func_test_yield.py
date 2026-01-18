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
def test_yield(self):
    a = ir.Yield(self.var_a, self.loc1, 0)
    b = ir.Yield(self.var_a, self.loc1, 0)
    c = ir.Yield(self.var_a, self.loc2, 0)
    d = ir.Yield(self.var_b, self.loc1, 0)
    e = ir.Yield(self.var_a, self.loc1, 1)
    self.check(a, same=[b, c], different=[d, e])