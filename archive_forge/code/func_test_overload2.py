import itertools
from numba.core import types
from numba.core.typeconv.typeconv import TypeManager, TypeCastingRules
from numba.core.typeconv import rules
from numba.core.typeconv import castgraph, Conversion
import unittest
def test_overload2(self):
    tm = rules.default_type_manager
    i16 = types.int16
    i32 = types.int32
    i64 = types.int64
    sig = (i32, i16, i32)
    ovs = [(i64, i64, i64), (i32, i32, i32), (i16, i16, i16)]
    self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=False, exact_match_required=False), 1)
    self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=True, exact_match_required=False), 1)
    ovs.reverse()
    self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=False, exact_match_required=False), 1)
    self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=True, exact_match_required=False), 1)