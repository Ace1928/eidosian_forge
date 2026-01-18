import pickle
import unittest
from traits.api import HasTraits, TraitError, PrefixList
def test_values_not_all_iterables(self):
    with self.assertRaises(TypeError) as exception_context:
        PrefixList('zero')
    self.assertEqual(str(exception_context.exception), "values should be a collection of strings, not 'zero'")