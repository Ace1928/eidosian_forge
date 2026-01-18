import unittest
from traits.api import HasTraits, Int, List
def test_pop_with_no_argument(self):
    foo = MyClass()
    item = foo.l.pop()
    self.assertEqual(item, 3)
    self.assertEqual(foo.l, [1, 2])
    self.assertEqual(len(foo.l_events), 1)
    event = foo.l_events[0]
    self.assertEqual(event.added, [])
    self.assertEqual(event.removed, [3])
    self.assertEqual(event.index, 2)