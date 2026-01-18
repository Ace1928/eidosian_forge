import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_on_trait_change_with_list_of_extended_names(self):
    dummy = Dummy(x=10)
    model = ExtendedListenerInList(dummy=dummy)
    self.assertFalse(model.changed)
    dummy.x = 11
    self.assertTrue(model.changed)