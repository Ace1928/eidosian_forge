import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_sort_reverse(self):
    f = Foo()
    f.l = ['a', 'c', 'b', 'd']
    f.l.sort(reverse=True)
    self.assertEqual(f.l, ['d', 'c', 'b', 'a'])