import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
class TestHasTraitsHelpersHasNamedTrait(unittest.TestCase):
    """ Test object_has_named_trait."""

    def test_object_has_named_trait_false_for_trait_list(self):
        foo = Foo()
        self.assertFalse(helpers.object_has_named_trait(foo.list_of_int, 'bar'), 'Expected object_has_named_trait to return false for {!r}'.format(type(foo.list_of_int)))

    def test_object_has_named_trait_true_basic(self):
        foo = Foo()
        self.assertTrue(helpers.object_has_named_trait(foo, 'list_of_int'), 'The named trait should exist.')

    def test_object_has_named_trait_false(self):
        foo = Foo()
        self.assertFalse(helpers.object_has_named_trait(foo, 'not_existing'), 'Expected object_has_named_trait to return False for anonexisting trait name.')

    def test_object_has_named_trait_does_not_trigger_property(self):
        foo = Foo()
        helpers.object_has_named_trait(foo, 'property_value')
        self.assertEqual(foo.property_n_calculations, 0)