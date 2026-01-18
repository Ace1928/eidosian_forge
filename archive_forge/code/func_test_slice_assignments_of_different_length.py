import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_slice_assignments_of_different_length(self):
    test_list = ['zero', 'one', 'two', 'three']
    f = Foo(l=test_list)
    f.l[1:3] = '01234'
    self.assertEqual(f.l, ['zero', '0', '1', '2', '3', '4', 'three'])
    f.l[4:] = []
    self.assertEqual(f.l, ['zero', '0', '1', '2'])
    f.l[:] = 'abcde'
    self.assertEqual(f.l, ['a', 'b', 'c', 'd', 'e'])
    f.l[:] = []
    self.assertEqual(f.l, [])
    f = Foo(l=test_list)
    with self.assertRaises(ValueError):
        f.l[::2] = ['a', 'b', 'c']
    self.assertEqual(f.l, test_list)
    with self.assertRaises(ValueError):
        f.l[::-1] = []
    self.assertEqual(f.l, test_list)