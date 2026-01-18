import pickle
import unittest
from traits.api import HasTraits, Int, PrefixMap, TraitError
def test_static_default_validation_error(self):
    with self.assertRaises(ValueError):

        class Person(HasTraits):
            married = PrefixMap({'yes': 1, 'yeah': 1, 'no': 0}, default_value='meh')