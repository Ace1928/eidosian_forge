import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class TraitWithMappingAndCallable(HasTraits):
    value = Trait('white', {'white': 0, 'red': 1, (0, 0, 0): 999}, str_cast_to_int)