import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_set_and_get_default_value(self):
    trait = CTrait(TraitKind.trait)
    trait.set_default_value(DefaultValue.constant, 2.3)
    self.assertEqual(trait.default_value(), (DefaultValue.constant, 2.3))
    trait.set_default_value(DefaultValue.list_copy, [1, 2, 3])
    self.assertEqual(trait.default_value(), (DefaultValue.list_copy, [1, 2, 3]))