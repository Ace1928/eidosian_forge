import pickle
import unittest
from traits.api import HasTraits, TraitError, PrefixList
def test_values_not_sequence(self):
    with self.assertRaises(TypeError):
        PrefixList('zero', 'one', 'two')