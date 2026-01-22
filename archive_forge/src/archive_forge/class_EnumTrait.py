import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class EnumTrait(HasTraits):
    value = Trait([1, 'one', 2, 'two', 3, 'three', 4.4, 'four.four'])