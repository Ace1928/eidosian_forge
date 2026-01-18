import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def test_static_notifiers_0(self):
    calls_0.clear()
    obj = StaticNotifiers0(ok=2)
    obj.ok = 3
    self.assertEqual(len(calls_0), 2)
    obj.fail = 1
    self.assertEqual(self.exceptions, [(obj, 'fail', 0, 1)])