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
class CheckEquality(unittest.TestCase):
    var_a = ir.Var(None, 'a', ir.unknown_loc)
    var_b = ir.Var(None, 'b', ir.unknown_loc)
    var_c = ir.Var(None, 'c', ir.unknown_loc)
    var_d = ir.Var(None, 'd', ir.unknown_loc)
    var_e = ir.Var(None, 'e', ir.unknown_loc)
    loc1 = ir.Loc('mock', 1, 0)
    loc2 = ir.Loc('mock', 2, 0)
    loc3 = ir.Loc('mock', 3, 0)

    def check(self, base, same=[], different=[]):
        for s in same:
            self.assertTrue(base == s)
        for d in different:
            self.assertTrue(base != d)