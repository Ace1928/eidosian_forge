import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_sort_key_reverse(self):
    f = Foo()
    f.l = ['a', 'c', 'b', 'd']
    f.l.sort(key=lambda x: -ord(x), reverse=True)
    self.assertEqual(f.l, ['a', 'b', 'c', 'd'])