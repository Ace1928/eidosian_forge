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
class TestMixedInts(TestCase):
    """
    Tests for operator calls with mixed integer types.
    """
    op = LiteralOperatorImpl
    int_samples = [0, 1, 3, 10, 42, 127, 10000, -1, -3, -10, -42, -127, -10000]
    int_types = [types.int8, types.uint8, types.int64, types.uint64]
    signed_types = [tp for tp in int_types if tp.signed]
    unsigned_types = [tp for tp in int_types if not tp.signed]
    type_pairs = list(itertools.product(int_types, int_types))
    signed_pairs = [(u, v) for u, v in type_pairs if u.signed or v.signed]
    unsigned_pairs = [(u, v) for u, v in type_pairs if not (u.signed or v.signed)]

    def get_numpy_signed_upcast(self, *vals):
        bitwidth = max((v.dtype.itemsize * 8 for v in vals))
        bitwidth = max(bitwidth, types.intp.bitwidth)
        return getattr(np, 'int%d' % bitwidth)

    def get_numpy_unsigned_upcast(self, *vals):
        bitwidth = max((v.dtype.itemsize * 8 for v in vals))
        bitwidth = max(bitwidth, types.intp.bitwidth)
        return getattr(np, 'uint%d' % bitwidth)

    def get_typed_int(self, typ, val):
        return getattr(np, typ.name)(val)

    def get_control_signed(self, opname):
        op = getattr(operator, opname)

        def control_signed(a, b):
            tp = self.get_numpy_signed_upcast(a, b)
            return op(tp(a), tp(b))
        return control_signed

    def get_control_unsigned(self, opname):
        op = getattr(operator, opname)

        def control_unsigned(a, b):
            tp = self.get_numpy_unsigned_upcast(a, b)
            return op(tp(a), tp(b))
        return control_unsigned

    def run_binary(self, pyfunc, control_func, operands, types, expected_type=int, force_type=lambda x: x, **assertPreciseEqualArgs):
        for xt, yt in types:
            cfunc = njit((xt, yt))(pyfunc)
            for x, y in itertools.product(operands, operands):
                x = self.get_typed_int(xt, x)
                y = self.get_typed_int(yt, y)
                expected = control_func(x, y)
                got = cfunc(x, y)
                self.assertIsInstance(got, expected_type)
                msg = 'mismatch for (%r, %r) with types %s' % (x, y, (xt, yt))
                got, expected = (force_type(got), force_type(expected))
                self.assertPreciseEqual(got, expected, msg=msg, **assertPreciseEqualArgs)

    def run_unary(self, pyfunc, control_func, operands, types, expected_type=int):
        for xt in types:
            cfunc = njit((xt,))(pyfunc)
            for x in operands:
                x = self.get_typed_int(xt, x)
                expected = control_func(x)
                got = cfunc(x)
                self.assertIsInstance(got, expected_type)
                self.assertPreciseEqual(got, expected, msg='mismatch for %r with type %s: %r != %r' % (x, xt, got, expected))

    def run_arith_binop(self, pyfunc, opname, samples, expected_type=int, force_type=lambda x: x, **assertPreciseEqualArgs):
        self.run_binary(pyfunc, self.get_control_signed(opname), samples, self.signed_pairs, expected_type, force_type=force_type, **assertPreciseEqualArgs)
        self.run_binary(pyfunc, self.get_control_unsigned(opname), samples, self.unsigned_pairs, expected_type, force_type=force_type, **assertPreciseEqualArgs)

    def test_add(self):
        self.run_arith_binop(self.op.add_usecase, 'add', self.int_samples)

    def test_sub(self):
        self.run_arith_binop(self.op.sub_usecase, 'sub', self.int_samples)

    def test_mul(self):
        self.run_arith_binop(self.op.mul_usecase, 'mul', self.int_samples)

    def test_floordiv(self):
        samples = [x for x in self.int_samples if x != 0]
        self.run_arith_binop(self.op.floordiv_usecase, 'floordiv', samples)

    def test_mod(self):
        samples = [x for x in self.int_samples if x != 0]
        self.run_arith_binop(self.op.mod_usecase, 'mod', samples)

    def test_pow(self):
        extra_cast = {}
        if utils.PYVERSION == (3, 11):
            extra_cast['force_type'] = float
        pyfunc = self.op.pow_usecase
        samples = [x for x in self.int_samples if x >= 0]
        self.run_arith_binop(pyfunc, 'pow', samples, **extra_cast)

        def control_signed(a, b):
            tp = self.get_numpy_signed_upcast(a, b)
            if b >= 0:
                return tp(a) ** tp(b)
            else:
                inv = tp(a) ** tp(-b)
                if inv == 0:
                    return 0
                return np.intp(1.0 / inv)
        samples = [x for x in self.int_samples if x != 0]
        signed_pairs = [(u, v) for u, v in self.type_pairs if u.signed and v.signed]
        self.run_binary(pyfunc, control_signed, samples, signed_pairs, **extra_cast)

    def test_truediv(self):

        def control(a, b):
            return float(a) / float(b)
        samples = [x for x in self.int_samples if x != 0]
        pyfunc = self.op.truediv_usecase
        self.run_binary(pyfunc, control, samples, self.signed_pairs, expected_type=float, prec='double')
        self.run_binary(pyfunc, control, samples, self.unsigned_pairs, expected_type=float, prec='double')

    def test_and(self):
        self.run_arith_binop(self.op.bitwise_and_usecase, 'and_', self.int_samples)

    def test_or(self):
        self.run_arith_binop(self.op.bitwise_or_usecase, 'or_', self.int_samples)

    def test_xor(self):
        self.run_arith_binop(self.op.bitwise_xor_usecase, 'xor', self.int_samples)

    def run_shift_binop(self, pyfunc, opname):
        opfunc = getattr(operator, opname)

        def control_signed(a, b):
            tp = self.get_numpy_signed_upcast(a, b)
            return opfunc(tp(a), tp(b))

        def control_unsigned(a, b):
            tp = self.get_numpy_unsigned_upcast(a, b)
            return opfunc(tp(a), tp(b))
        samples = self.int_samples

        def check(xt, yt, control_func):
            cfunc = njit((xt, yt))(pyfunc)
            for x in samples:
                maxshift = xt.bitwidth - 1
                for y in (0, 1, 3, 5, maxshift - 1, maxshift):
                    x = self.get_typed_int(xt, x)
                    y = self.get_typed_int(yt, y)
                    expected = control_func(x, y)
                    got = cfunc(x, y)
                    msg = 'mismatch for (%r, %r) with types %s' % (x, y, (xt, yt))
                    self.assertPreciseEqual(got, expected, msg=msg)
        signed_pairs = [(u, v) for u, v in self.type_pairs if u.signed]
        unsigned_pairs = [(u, v) for u, v in self.type_pairs if not u.signed]
        for xt, yt in signed_pairs:
            check(xt, yt, control_signed)
        for xt, yt in unsigned_pairs:
            check(xt, yt, control_unsigned)

    def test_lshift(self):
        self.run_shift_binop(self.op.bitshift_left_usecase, 'lshift')

    def test_rshift(self):
        self.run_shift_binop(self.op.bitshift_right_usecase, 'rshift')

    def test_unary_positive(self):

        def control(a):
            return a
        samples = self.int_samples
        pyfunc = self.op.unary_positive_usecase
        self.run_unary(pyfunc, control, samples, self.int_types)

    def test_unary_negative(self):

        def control_signed(a):
            tp = self.get_numpy_signed_upcast(a)
            return tp(-a)

        def control_unsigned(a):
            tp = self.get_numpy_unsigned_upcast(a)
            return tp(-a)
        samples = self.int_samples
        pyfunc = self.op.negate_usecase
        self.run_unary(pyfunc, control_signed, samples, self.signed_types)
        self.run_unary(pyfunc, control_unsigned, samples, self.unsigned_types)

    def test_invert(self):

        def control_signed(a):
            tp = self.get_numpy_signed_upcast(a)
            return tp(~a)

        def control_unsigned(a):
            tp = self.get_numpy_unsigned_upcast(a)
            return tp(~a)
        samples = self.int_samples
        pyfunc = self.op.bitwise_not_usecase
        self.run_unary(pyfunc, control_signed, samples, self.signed_types)
        self.run_unary(pyfunc, control_unsigned, samples, self.unsigned_types)