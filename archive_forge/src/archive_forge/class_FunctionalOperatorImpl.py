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
class FunctionalOperatorImpl(object):

    @staticmethod
    def add_usecase(x, y):
        return operator.add(x, y)

    @staticmethod
    def iadd_usecase(x, y):
        return operator.iadd(x, y)

    @staticmethod
    def sub_usecase(x, y):
        return operator.sub(x, y)

    @staticmethod
    def isub_usecase(x, y):
        return operator.isub(x, y)

    @staticmethod
    def mul_usecase(x, y):
        return operator.mul(x, y)

    @staticmethod
    def imul_usecase(x, y):
        return operator.imul(x, y)

    @staticmethod
    def floordiv_usecase(x, y):
        return operator.floordiv(x, y)

    @staticmethod
    def ifloordiv_usecase(x, y):
        return operator.ifloordiv(x, y)

    @staticmethod
    def truediv_usecase(x, y):
        return operator.truediv(x, y)

    @staticmethod
    def itruediv_usecase(x, y):
        return operator.itruediv(x, y)

    @staticmethod
    def mod_usecase(x, y):
        return operator.mod(x, y)

    @staticmethod
    def imod_usecase(x, y):
        return operator.imod(x, y)

    @staticmethod
    def pow_usecase(x, y):
        return operator.pow(x, y)

    @staticmethod
    def ipow_usecase(x, y):
        return operator.ipow(x, y)

    @staticmethod
    def matmul_usecase(x, y):
        return operator.matmul(x, y)

    @staticmethod
    def imatmul_usecase(x, y):
        return operator.imatmul(x, y)

    @staticmethod
    def bitshift_left_usecase(x, y):
        return operator.lshift(x, y)

    @staticmethod
    def bitshift_ileft_usecase(x, y):
        return operator.ilshift(x, y)

    @staticmethod
    def bitshift_right_usecase(x, y):
        return operator.rshift(x, y)

    @staticmethod
    def bitshift_iright_usecase(x, y):
        return operator.irshift(x, y)

    @staticmethod
    def bitwise_and_usecase(x, y):
        return operator.and_(x, y)

    @staticmethod
    def bitwise_iand_usecase(x, y):
        return operator.iand(x, y)

    @staticmethod
    def bitwise_or_usecase(x, y):
        return operator.or_(x, y)

    @staticmethod
    def bitwise_ior_usecase(x, y):
        return operator.ior(x, y)

    @staticmethod
    def bitwise_xor_usecase(x, y):
        return operator.xor(x, y)

    @staticmethod
    def bitwise_ixor_usecase(x, y):
        return operator.ixor(x, y)

    @staticmethod
    def bitwise_not_usecase_binary(x, _unused):
        return operator.invert(x)

    @staticmethod
    def bitwise_not_usecase(x):
        return operator.invert(x)

    @staticmethod
    def not_usecase(x):
        return operator.not_(x)

    @staticmethod
    def negate_usecase(x):
        return operator.neg(x)

    @staticmethod
    def unary_positive_usecase(x):
        return operator.pos(x)

    @staticmethod
    def lt_usecase(x, y):
        return operator.lt(x, y)

    @staticmethod
    def le_usecase(x, y):
        return operator.le(x, y)

    @staticmethod
    def gt_usecase(x, y):
        return operator.gt(x, y)

    @staticmethod
    def ge_usecase(x, y):
        return operator.ge(x, y)

    @staticmethod
    def eq_usecase(x, y):
        return operator.eq(x, y)

    @staticmethod
    def ne_usecase(x, y):
        return operator.ne(x, y)

    @staticmethod
    def in_usecase(x, y):
        return operator.contains(y, x)

    @staticmethod
    def not_in_usecase(x, y):
        return not operator.contains(y, x)

    @staticmethod
    def is_usecase(x, y):
        return operator.is_(x, y)