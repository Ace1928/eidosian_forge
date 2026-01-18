import unittest
from traits.api import HasTraits, Str, Int
from traits.testing.unittest_tools import UnittestTools
def test_trait_get(self):
    obj = TraitsObject()
    obj.trait_set(string='foo')
    values = obj.trait_get('string', 'integer')
    self.assertEqual(values, {'string': 'foo', 'integer': 0})