import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class HasComparisonMode(HasTraits):
    bar = Trait(comparison_mode=ComparisonMode.equality)