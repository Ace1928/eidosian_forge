import unittest
from traits.api import ArrayOrNone, ComparisonMode, HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_setting_array_from_none(self):
    foo = Foo()
    test_array = numpy.arange(5)
    self.assertIsNone(foo.maybe_array)
    foo.maybe_array = test_array
    self.assertIsInstance(foo.maybe_array, numpy.ndarray)
    foo.maybe_array = None
    self.assertIsNone(foo.maybe_array)