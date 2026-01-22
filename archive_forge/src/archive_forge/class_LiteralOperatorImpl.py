import copy
import itertools
import operator
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, utils, errors
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase, tag, needs_blas
from numba.tests.matmul_usecase import (matmul_usecase, imatmul_usecase,
class LiteralOperatorImpl(object):

    @staticmethod
    def add_usecase(x, y):
        return x + y

    @staticmethod
    def iadd_usecase(x, y):
        x += y
        return x

    @staticmethod
    def sub_usecase(x, y):
        return x - y

    @staticmethod
    def isub_usecase(x, y):
        x -= y
        return x

    @staticmethod
    def mul_usecase(x, y):
        return x * y

    @staticmethod
    def imul_usecase(x, y):
        x *= y
        return x

    @staticmethod
    def floordiv_usecase(x, y):
        return x // y

    @staticmethod
    def ifloordiv_usecase(x, y):
        x //= y
        return x

    @staticmethod
    def truediv_usecase(x, y):
        return x / y

    @staticmethod
    def itruediv_usecase(x, y):
        x /= y
        return x
    if matmul_usecase:
        matmul_usecase = staticmethod(matmul_usecase)
        imatmul_usecase = staticmethod(imatmul_usecase)

    @staticmethod
    def mod_usecase(x, y):
        return x % y

    @staticmethod
    def imod_usecase(x, y):
        x %= y
        return x

    @staticmethod
    def pow_usecase(x, y):
        return x ** y

    @staticmethod
    def ipow_usecase(x, y):
        x **= y
        return x

    @staticmethod
    def bitshift_left_usecase(x, y):
        return x << y

    @staticmethod
    def bitshift_ileft_usecase(x, y):
        x <<= y
        return x

    @staticmethod
    def bitshift_right_usecase(x, y):
        return x >> y

    @staticmethod
    def bitshift_iright_usecase(x, y):
        x >>= y
        return x

    @staticmethod
    def bitwise_and_usecase(x, y):
        return x & y

    @staticmethod
    def bitwise_iand_usecase(x, y):
        x &= y
        return x

    @staticmethod
    def bitwise_or_usecase(x, y):
        return x | y

    @staticmethod
    def bitwise_ior_usecase(x, y):
        x |= y
        return x

    @staticmethod
    def bitwise_xor_usecase(x, y):
        return x ^ y

    @staticmethod
    def bitwise_ixor_usecase(x, y):
        x ^= y
        return x

    @staticmethod
    def bitwise_not_usecase_binary(x, _unused):
        return ~x

    @staticmethod
    def bitwise_not_usecase(x):
        return ~x

    @staticmethod
    def not_usecase(x):
        return not x

    @staticmethod
    def negate_usecase(x):
        return -x

    @staticmethod
    def unary_positive_usecase(x):
        return +x

    @staticmethod
    def lt_usecase(x, y):
        return x < y

    @staticmethod
    def le_usecase(x, y):
        return x <= y

    @staticmethod
    def gt_usecase(x, y):
        return x > y

    @staticmethod
    def ge_usecase(x, y):
        return x >= y

    @staticmethod
    def eq_usecase(x, y):
        return x == y

    @staticmethod
    def ne_usecase(x, y):
        return x != y

    @staticmethod
    def in_usecase(x, y):
        return x in y

    @staticmethod
    def not_in_usecase(x, y):
        return x not in y

    @staticmethod
    def is_usecase(x, y):
        return x is y