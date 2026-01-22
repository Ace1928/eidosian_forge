import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class ImaginaryValueTrait(HasTraits):
    value = Trait(99.0 - 99j)