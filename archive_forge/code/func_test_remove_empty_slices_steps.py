import unittest
from traits.api import HasTraits, Int, List
def test_remove_empty_slices_steps(self):
    foo = MyClass()
    del foo.l[3::2]
    self.assertEqual(foo.l, [1, 2, 3])
    self.assertEqual(len(foo.l_events), 0)