import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def value_set(self, value):
    old_value = self.__dict__.get('_value', 0)
    if value != old_value:
        self._value = value
        self.trait_property_changed('value', old_value, value)