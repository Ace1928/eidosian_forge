import unittest
from traits.api import HasTraits, Int, List
def test_pop_out_of_range(self):
    foo = MyClass()
    with self.assertRaises(IndexError):
        foo.l.pop(-4)
    with self.assertRaises(IndexError):
        foo.l.pop(3)
    self.assertEqual(foo.l, [1, 2, 3])
    self.assertEqual(len(foo.l_events), 0)