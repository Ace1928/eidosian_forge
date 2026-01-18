import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def test_object_has_named_trait_true_basic(self):
    foo = Foo()
    self.assertTrue(helpers.object_has_named_trait(foo, 'list_of_int'), 'The named trait should exist.')