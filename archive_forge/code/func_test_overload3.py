import itertools
from numba.core import types
from numba.core.typeconv.typeconv import TypeManager, TypeCastingRules
from numba.core.typeconv import rules
from numba.core.typeconv import castgraph, Conversion
import unittest
def test_overload3(self):
    tm = rules.default_type_manager
    i32 = types.int32
    i64 = types.int64
    f64 = types.float64
    sig = (i32, i32)
    ovs = [(i64, i64), (f64, f64)]
    self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=False, exact_match_required=False), 0)
    self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=True, exact_match_required=False), 0)
    ovs.reverse()
    self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=False, exact_match_required=False), 1)
    self.assertEqual(tm.select_overload(sig, ovs, allow_unsafe=True, exact_match_required=False), 1)