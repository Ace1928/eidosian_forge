import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def test_object_has_named_trait_false_for_trait_list(self):
    foo = Foo()
    self.assertFalse(helpers.object_has_named_trait(foo.list_of_int, 'bar'), 'Expected object_has_named_trait to return false for {!r}'.format(type(foo.list_of_int)))