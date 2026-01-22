import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
class TestHasTraitsHelpersIterObjects(unittest.TestCase):
    """ Test iter_objects."""

    def test_iter_objects_avoid_undefined(self):
        foo = Foo()
        self.assertNotIn('instance', foo.__dict__)
        actual = list(helpers.iter_objects(foo, 'instance'))
        self.assertEqual(actual, [])

    def test_iter_objects_no_sideeffect(self):
        foo = Foo()
        self.assertNotIn('instance', foo.__dict__)
        list(helpers.iter_objects(foo, 'instance'))
        self.assertNotIn('instance', foo.__dict__)

    def test_iter_objects_avoid_none(self):
        foo = Foo()
        foo.instance = None
        actual = list(helpers.iter_objects(foo, 'instance'))
        self.assertEqual(actual, [])

    def test_iter_objects_allow_object_cannot_compare_for_equality(self):
        foo = Foo()
        foo.any_value = CannotCompare()
        actual = list(helpers.iter_objects(foo, 'any_value'))
        self.assertEqual(actual, [foo.any_value])

    def test_iter_objects_accepted_values(self):
        foo = Foo(instance=Bar(), list_of_int=[1, 2])
        actual = list(helpers.iter_objects(foo, 'instance'))
        self.assertEqual(actual, [foo.instance])

    def test_iter_objects_does_not_evaluate_default(self):
        foo = Foo()
        list(helpers.iter_objects(foo, 'int_with_default'))
        self.assertFalse(foo.int_with_default_computed, 'Expect the default not to have been computed.')

    def test_iter_objects_does_not_trigger_property(self):
        foo = Foo()
        list(helpers.iter_objects(foo, 'property_value'))
        self.assertEqual(foo.property_n_calculations, 0)