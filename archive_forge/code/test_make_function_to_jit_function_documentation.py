from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest

    This tests the pass that converts ir.Expr.op == make_function (i.e. closure)
    into a JIT function.
    