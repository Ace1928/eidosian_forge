import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class IntTrait(HasTraits):
    value = Int(99)