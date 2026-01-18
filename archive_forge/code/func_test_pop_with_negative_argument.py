import unittest
from traits.api import HasTraits, Int, List
def test_pop_with_negative_argument(self):
    foo = MyClass()
    item = foo.l.pop(-2)
    self.assertEqual(item, 2)
    self.assertEqual(foo.l, [1, 3])
    self.assertEqual(len(foo.l_events), 1)
    event = foo.l_events[0]
    self.assertEqual(event.added, [])
    self.assertEqual(event.removed, [2])
    self.assertEqual(event.index, 1)