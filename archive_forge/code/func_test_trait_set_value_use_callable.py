import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_trait_set_value_use_callable(self):
    obj = TraitWithMappingAndCallable(value='red')
    self.assertEqual(obj.value, 3)
    self.assertEqual(obj.value_, 3)