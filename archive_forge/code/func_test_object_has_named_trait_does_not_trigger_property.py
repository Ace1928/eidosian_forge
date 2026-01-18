import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def test_object_has_named_trait_does_not_trigger_property(self):
    foo = Foo()
    helpers.object_has_named_trait(foo, 'property_value')
    self.assertEqual(foo.property_n_calculations, 0)