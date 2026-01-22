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
class TestUnify(unittest.TestCase):
    """
    Tests for type unification with a typing context.
    """
    int_unify = {('uint8', 'uint8'): 'uint8', ('int8', 'int8'): 'int8', ('uint16', 'uint16'): 'uint16', ('int16', 'int16'): 'int16', ('uint32', 'uint32'): 'uint32', ('int32', 'int32'): 'int32', ('uint64', 'uint64'): 'uint64', ('int64', 'int64'): 'int64', ('int8', 'uint8'): 'int16', ('int8', 'uint16'): 'int32', ('int8', 'uint32'): 'int64', ('uint8', 'int32'): 'int32', ('uint8', 'uint64'): 'uint64', ('int16', 'int8'): 'int16', ('int16', 'uint8'): 'int16', ('int16', 'uint16'): 'int32', ('int16', 'uint32'): 'int64', ('int16', 'int64'): 'int64', ('int16', 'uint64'): 'float64', ('uint16', 'uint8'): 'uint16', ('uint16', 'uint32'): 'uint32', ('uint16', 'int32'): 'int32', ('uint16', 'uint64'): 'uint64', ('int32', 'int8'): 'int32', ('int32', 'int16'): 'int32', ('int32', 'uint32'): 'int64', ('int32', 'int64'): 'int64', ('uint32', 'uint8'): 'uint32', ('uint32', 'int64'): 'int64', ('uint32', 'uint64'): 'uint64', ('int64', 'int8'): 'int64', ('int64', 'uint8'): 'int64', ('int64', 'uint16'): 'int64', ('uint64', 'int8'): 'float64', ('uint64', 'int32'): 'float64', ('uint64', 'int64'): 'float64'}

    def assert_unify(self, aty, bty, expected):
        ctx = typing.Context()
        template = '{0}, {1} -> {2} != {3}'
        for unify_func in (ctx.unify_types, ctx.unify_pairs):
            unified = unify_func(aty, bty)
            self.assertEqual(unified, expected, msg=template.format(aty, bty, unified, expected))
            unified = unify_func(bty, aty)
            self.assertEqual(unified, expected, msg=template.format(bty, aty, unified, expected))

    def assert_unify_failure(self, aty, bty):
        self.assert_unify(aty, bty, None)

    def test_integer(self):
        ctx = typing.Context()
        for aty, bty in itertools.product(types.integer_domain, types.integer_domain):
            key = (str(aty), str(bty))
            try:
                expected = self.int_unify[key]
            except KeyError:
                expected = self.int_unify[key[::-1]]
            self.assert_unify(aty, bty, getattr(types, expected))

    def test_bool(self):
        aty = types.boolean
        for bty in types.integer_domain:
            self.assert_unify(aty, bty, bty)
        for cty in types.real_domain:
            self.assert_unify(aty, cty, cty)

    def unify_number_pair_test(self, n):
        """
        Test all permutations of N-combinations of numeric types and ensure
        that the order of types in the sequence is irrelevant.
        """
        ctx = typing.Context()
        for tys in itertools.combinations(types.number_domain, n):
            res = [ctx.unify_types(*comb) for comb in itertools.permutations(tys)]
            first_result = res[0]
            self.assertIsInstance(first_result, types.Number)
            for other in res[1:]:
                self.assertEqual(first_result, other)

    def test_unify_number_pair(self):
        self.unify_number_pair_test(2)
        self.unify_number_pair_test(3)

    def test_none_to_optional(self):
        """
        Test unification of `none` and multiple number types to optional type
        """
        ctx = typing.Context()
        for tys in itertools.combinations(types.number_domain, 2):
            tys = list(tys)
            expected = types.Optional(ctx.unify_types(*tys))
            results = [ctx.unify_types(*comb) for comb in itertools.permutations(tys + [types.none])]
            for res in results:
                self.assertEqual(res, expected)

    def test_none(self):
        aty = types.none
        bty = types.none
        self.assert_unify(aty, bty, types.none)

    def test_optional(self):
        aty = types.Optional(i32)
        bty = types.none
        self.assert_unify(aty, bty, aty)
        aty = types.Optional(i32)
        bty = types.Optional(i64)
        self.assert_unify(aty, bty, bty)
        aty = types.Optional(i32)
        bty = i64
        self.assert_unify(aty, bty, types.Optional(i64))
        aty = types.Optional(i32)
        bty = types.Optional(types.slice3_type)
        self.assert_unify_failure(aty, bty)

    def test_tuple(self):
        aty = types.UniTuple(i32, 3)
        bty = types.UniTuple(i64, 3)
        self.assert_unify(aty, bty, types.UniTuple(i64, 3))
        aty = types.UniTuple(i32, 2)
        bty = types.Tuple((i16, i64))
        self.assert_unify(aty, bty, types.Tuple((i32, i64)))
        aty = types.UniTuple(i64, 0)
        bty = types.Tuple(())
        self.assert_unify(aty, bty, bty)
        aty = types.Tuple((i8, i16, i32))
        bty = types.Tuple((i32, i16, i8))
        self.assert_unify(aty, bty, types.Tuple((i32, i16, i32)))
        aty = types.Tuple((i8, i32))
        bty = types.Tuple((i32, i8))
        self.assert_unify(aty, bty, types.Tuple((i32, i32)))
        aty = types.Tuple((i8, i16))
        bty = types.Tuple((i16, i8))
        self.assert_unify(aty, bty, types.Tuple((i16, i16)))
        aty = types.UniTuple(f64, 3)
        bty = types.UniTuple(c64, 3)
        self.assert_unify(aty, bty, types.UniTuple(c128, 3))
        aty = types.UniTuple(types.Tuple((u32, f32)), 2)
        bty = types.UniTuple(types.Tuple((i16, f32)), 2)
        self.assert_unify(aty, bty, types.UniTuple(types.Tuple((i64, f32)), 2))
        aty = types.UniTuple(i32, 1)
        bty = types.UniTuple(types.slice3_type, 1)
        self.assert_unify_failure(aty, bty)
        aty = types.UniTuple(i32, 1)
        bty = types.UniTuple(i32, 2)
        self.assert_unify_failure(aty, bty)
        aty = types.Tuple((i8, types.slice3_type))
        bty = types.Tuple((i32, i8))
        self.assert_unify_failure(aty, bty)

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

    def test_arrays(self):
        aty = types.Array(i32, 3, 'C')
        bty = types.Array(i32, 3, 'A')
        self.assert_unify(aty, bty, bty)
        aty = types.Array(i32, 3, 'C')
        bty = types.Array(i32, 3, 'F')
        self.assert_unify(aty, bty, types.Array(i32, 3, 'A'))
        aty = types.Array(i32, 3, 'C')
        bty = types.Array(i32, 3, 'C', readonly=True)
        self.assert_unify(aty, bty, bty)
        aty = types.Array(i32, 3, 'A')
        bty = types.Array(i32, 3, 'C', readonly=True)
        self.assert_unify(aty, bty, types.Array(i32, 3, 'A', readonly=True))
        aty = types.Array(i32, 2, 'C')
        bty = types.Array(i32, 3, 'C')
        self.assert_unify_failure(aty, bty)
        aty = types.Array(i32, 2, 'C')
        bty = types.Array(u32, 2, 'C')
        self.assert_unify_failure(aty, bty)

    def test_list(self):
        aty = types.List(types.undefined)
        bty = types.List(i32)
        self.assert_unify(aty, bty, bty)
        aty = types.List(i16)
        bty = types.List(i32)
        self.assert_unify(aty, bty, bty)
        aty = types.List(types.Tuple([i32, i16]))
        bty = types.List(types.Tuple([i16, i64]))
        cty = types.List(types.Tuple([i32, i64]))
        self.assert_unify(aty, bty, cty)
        aty = types.List(i16, reflected=True)
        bty = types.List(i32)
        cty = types.List(i32, reflected=True)
        self.assert_unify(aty, bty, cty)
        aty = types.List(i16)
        bty = types.List(types.Tuple([i16]))
        self.assert_unify_failure(aty, bty)

    def test_set(self):
        aty = types.Set(i16, reflected=True)
        bty = types.Set(i32)
        cty = types.Set(i32, reflected=True)
        self.assert_unify(aty, bty, cty)
        aty = types.Set(i16)
        bty = types.Set(types.Tuple([i16]))
        self.assert_unify_failure(aty, bty)

    def test_range(self):
        aty = types.range_state32_type
        bty = types.range_state64_type
        self.assert_unify(aty, bty, bty)