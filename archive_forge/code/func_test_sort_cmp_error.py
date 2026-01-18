import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_sort_cmp_error(self):
    f = Foo()
    f.l = ['a', 'c', 'b', 'd']
    with self.assertRaises(TypeError):
        f.l.sort(cmp=lambda x, y: ord(x) - ord(y))