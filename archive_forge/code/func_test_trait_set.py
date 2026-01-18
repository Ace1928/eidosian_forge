import unittest
from traits.api import HasTraits, Str, Int
from traits.testing.unittest_tools import UnittestTools
def test_trait_set(self):
    obj = TraitsObject()
    obj.trait_set(string='foo')
    self.assertEqual(obj.string, 'foo')
    self.assertEqual(obj.integer, 0)