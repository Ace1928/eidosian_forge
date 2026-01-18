import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def test_static_notifiers_2(self):
    obj = StaticNotifiers2(ok=2)
    obj.ok = 3
    self.assertEqual(obj.calls, [2, 3])
    obj.fail = 1
    self.assertEqual(self.exceptions, [(obj, 'fail', 0, 1)])