import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def test_static_notifiers_4(self):
    obj = StaticNotifiers4(ok=2)
    obj.ok = 3
    self.assertEqual(obj.calls, [('ok', 0, 2), ('ok', 2, 3)])
    obj.fail = 1
    self.assertEqual(self.exceptions, [(obj, 'fail', 0, 1)])