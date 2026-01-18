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
def test_terminator(self):
    t1 = ir.Terminator()
    t2 = ir.Terminator()
    self.check(t1, same=[t2])