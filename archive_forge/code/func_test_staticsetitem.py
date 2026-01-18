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
def test_staticsetitem(self):
    a = ir.StaticSetItem(self.var_a, 1, self.var_b, self.var_c, self.loc1)
    b = ir.StaticSetItem(self.var_a, 1, self.var_b, self.var_c, self.loc1)
    c = ir.StaticSetItem(self.var_a, 1, self.var_b, self.var_c, self.loc2)
    d = ir.StaticSetItem(self.var_d, 1, self.var_b, self.var_c, self.loc1)
    e = ir.StaticSetItem(self.var_a, 2, self.var_b, self.var_c, self.loc1)
    f = ir.StaticSetItem(self.var_a, 1, self.var_d, self.var_c, self.loc1)
    g = ir.StaticSetItem(self.var_a, 1, self.var_b, self.var_d, self.loc1)
    self.check(a, same=[b, c], different=[d, e, f, g])