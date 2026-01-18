import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def test_static_notifiers_3(self):
    obj = StaticNotifiers3(ok=2)
    obj.ok = 3
    self.assertEqual(obj.calls, [(0, 2), (2, 3)])
    obj.fail = 1
    self.assertEqual(self.exceptions, [(obj, 'fail', 0, 1)])