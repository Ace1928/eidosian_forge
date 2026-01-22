import itertools
from numba.core import types
from numba.core.typeconv.typeconv import TypeManager, TypeCastingRules
from numba.core.typeconv import rules
from numba.core.typeconv import castgraph, Conversion
import unittest
class CompatibilityTestMixin(unittest.TestCase):

    def check_number_compatibility(self, check_compatible):
        b = types.boolean
        i8 = types.int8
        i16 = types.int16
        i32 = types.int32
        i64 = types.int64
        u8 = types.uint8
        u16 = types.uint16
        u32 = types.uint32
        u64 = types.uint64
        f16 = types.float16
        f32 = types.float32
        f64 = types.float64
        c64 = types.complex64
        c128 = types.complex128
        self.assertEqual(check_compatible(i32, i32), Conversion.exact)
        self.assertEqual(check_compatible(b, i8), Conversion.safe)
        self.assertEqual(check_compatible(b, u8), Conversion.safe)
        self.assertEqual(check_compatible(i8, b), Conversion.unsafe)
        self.assertEqual(check_compatible(u8, b), Conversion.unsafe)
        self.assertEqual(check_compatible(i32, i64), Conversion.promote)
        self.assertEqual(check_compatible(i32, u32), Conversion.unsafe)
        self.assertEqual(check_compatible(u32, i32), Conversion.unsafe)
        self.assertEqual(check_compatible(u32, i64), Conversion.safe)
        self.assertEqual(check_compatible(i16, f16), Conversion.unsafe)
        self.assertEqual(check_compatible(i32, f32), Conversion.unsafe)
        self.assertEqual(check_compatible(u32, f32), Conversion.unsafe)
        self.assertEqual(check_compatible(i32, f64), Conversion.safe)
        self.assertEqual(check_compatible(u32, f64), Conversion.safe)
        self.assertEqual(check_compatible(i64, f64), Conversion.safe)
        self.assertEqual(check_compatible(u64, f64), Conversion.safe)
        self.assertEqual(check_compatible(f32, c64), Conversion.safe)
        self.assertEqual(check_compatible(f64, c128), Conversion.safe)
        self.assertEqual(check_compatible(f64, c64), Conversion.unsafe)
        self.assertEqual(check_compatible(i16, f64), Conversion.safe)
        self.assertEqual(check_compatible(i16, i64), Conversion.promote)
        self.assertEqual(check_compatible(i32, c64), Conversion.unsafe)
        self.assertEqual(check_compatible(i32, c128), Conversion.safe)
        self.assertEqual(check_compatible(i32, u64), Conversion.unsafe)
        for ta, tb in itertools.product(types.number_domain, types.number_domain):
            if ta in types.complex_domain and tb not in types.complex_domain:
                continue
            self.assertTrue(check_compatible(ta, tb) is not None, msg='No cast from %s to %s' % (ta, tb))