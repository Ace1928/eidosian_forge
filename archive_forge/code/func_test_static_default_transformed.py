import pickle
import unittest
from traits.api import HasTraits, Int, PrefixMap, TraitError
def test_static_default_transformed(self):

    class Person(HasTraits):
        married = PrefixMap({'yes': 1, 'yeah': 1, 'no': 0}, default_value='yea')
    p = Person()
    self.assertEqual(p.married, 'yeah')
    self.assertEqual(p.married_, 1)
    p = Person()
    self.assertEqual(p.married_, 1)
    self.assertEqual(p.married, 'yeah')