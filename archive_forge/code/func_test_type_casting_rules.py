import itertools
from numba.core import types
from numba.core.typeconv.typeconv import TypeManager, TypeCastingRules
from numba.core.typeconv import rules
from numba.core.typeconv import castgraph, Conversion
import unittest
def test_type_casting_rules(self):
    tm = TypeManager()
    tcr = TypeCastingRules(tm)
    i16 = types.int16
    i32 = types.int32
    i64 = types.int64
    f64 = types.float64
    f32 = types.float32
    f16 = types.float16
    made_up = types.Dummy('made_up')
    tcr.promote_unsafe(i32, i64)
    tcr.safe_unsafe(i32, f64)
    tcr.promote_unsafe(f32, f64)
    tcr.promote_unsafe(f16, f32)
    tcr.unsafe_unsafe(i16, f16)

    def base_test():
        self.assertEqual(tm.check_compatible(i32, i64), Conversion.promote)
        self.assertEqual(tm.check_compatible(i32, f64), Conversion.safe)
        self.assertEqual(tm.check_compatible(f16, f32), Conversion.promote)
        self.assertEqual(tm.check_compatible(f32, f64), Conversion.promote)
        self.assertEqual(tm.check_compatible(i64, i32), Conversion.unsafe)
        self.assertEqual(tm.check_compatible(f64, i32), Conversion.unsafe)
        self.assertEqual(tm.check_compatible(f64, f32), Conversion.unsafe)
        self.assertEqual(tm.check_compatible(i64, f64), Conversion.unsafe)
        self.assertEqual(tm.check_compatible(f64, i64), Conversion.unsafe)
        self.assertEqual(tm.check_compatible(i64, f32), Conversion.unsafe)
        self.assertEqual(tm.check_compatible(i32, f32), Conversion.unsafe)
        self.assertEqual(tm.check_compatible(f32, i32), Conversion.unsafe)
        self.assertEqual(tm.check_compatible(i16, f16), Conversion.unsafe)
        self.assertEqual(tm.check_compatible(f16, i16), Conversion.unsafe)
    base_test()
    self.assertIsNone(tm.check_compatible(i64, made_up))
    self.assertIsNone(tm.check_compatible(i32, made_up))
    self.assertIsNone(tm.check_compatible(f32, made_up))
    self.assertIsNone(tm.check_compatible(made_up, f64))
    self.assertIsNone(tm.check_compatible(made_up, i64))
    tcr.promote(f64, made_up)
    tcr.unsafe(made_up, i32)
    base_test()
    self.assertEqual(tm.check_compatible(i64, made_up), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(i32, made_up), Conversion.safe)
    self.assertEqual(tm.check_compatible(f32, made_up), Conversion.promote)
    self.assertEqual(tm.check_compatible(made_up, f64), Conversion.unsafe)
    self.assertEqual(tm.check_compatible(made_up, i64), Conversion.unsafe)