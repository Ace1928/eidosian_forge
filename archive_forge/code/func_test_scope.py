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
def test_scope(self):
    parent1 = ir.Scope(None, self.loc1)
    parent2 = ir.Scope(None, self.loc1)
    parent3 = ir.Scope(None, self.loc2)
    self.check(parent1, same=[parent2, parent3])
    a = ir.Scope(parent1, self.loc1)
    b = ir.Scope(parent1, self.loc1)
    c = ir.Scope(parent1, self.loc2)
    d = ir.Scope(parent3, self.loc1)
    self.check(a, same=[b, c, d])
    e = ir.Scope(parent2, self.loc1)
    self.check(a, same=[e])