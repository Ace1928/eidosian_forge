import pickle
import pyomo.common.unittest as unittest
from pyomo.core.base.range import (
from pyomo.core.base.set import Any
def test_range_intersection(self):
    a = NNR('a')
    b = NR(0, 5, 0)
    c = NR(5, 10, 1)
    x = RP([[a], [b, c]])
    y = RP([[a], [c]])
    z = RP([[a], [b], [c]])
    w = RP([list(Any.ranges()), [b]])
    self.assertEqual(x.range_intersection([x]), [x])
    self.assertEqual(x.range_intersection([y]), [y])
    self.assertEqual(x.range_intersection([z]), [])
    self.assertEqual(x.range_intersection(Any.ranges()), [x])
    self.assertEqual(x.range_intersection([w]), [RP([[a], [b]])])
    self.assertEqual(y.range_intersection([w]), [RP([[a], [NR(5, 5, 0)]])])
    v = RP([[AnyRange()], [NR(0, 5, 0, (False, False))]])
    self.assertEqual(y.range_intersection([v]), [])