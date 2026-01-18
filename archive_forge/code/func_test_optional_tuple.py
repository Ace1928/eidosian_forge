import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
def test_optional_tuple(self):
    aty = types.none
    bty = types.UniTuple(i32, 2)
    self.assert_unify(aty, bty, types.Optional(types.UniTuple(i32, 2)))
    aty = types.Optional(types.UniTuple(i16, 2))
    bty = types.UniTuple(i32, 2)
    self.assert_unify(aty, bty, types.Optional(types.UniTuple(i32, 2)))
    aty = types.Tuple((types.none, i32))
    bty = types.Tuple((i16, types.none))
    self.assert_unify(aty, bty, types.Tuple((types.Optional(i16), types.Optional(i32))))
    aty = types.Tuple((types.Optional(i32), i64))
    bty = types.Tuple((i16, types.Optional(i8)))
    self.assert_unify(aty, bty, types.Tuple((types.Optional(i32), types.Optional(i64))))