import unittest
from traits.api import HasTraits, Int, List
def test_remove_item_not_present(self):
    foo = MyClass()
    with self.assertRaises(ValueError):
        foo.l.remove(1729)
    self.assertEqual(foo.l, [1, 2, 3])
    self.assertEqual(len(foo.l_events), 0)