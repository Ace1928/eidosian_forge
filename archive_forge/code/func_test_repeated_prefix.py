import pickle
import unittest
from traits.api import HasTraits, TraitError, PrefixList
def test_repeated_prefix(self):

    class A(HasTraits):
        foo = PrefixList(('abc1', 'abc2'))
    a = A()
    a.foo = 'abc1'
    self.assertEqual(a.foo, 'abc1')
    with self.assertRaises(TraitError):
        a.foo = 'abc'