import unittest
import warnings
from traits.api import (
class RichCompare(HasTraits):
    bar = Any(comparison_mode=ComparisonMode.equality)