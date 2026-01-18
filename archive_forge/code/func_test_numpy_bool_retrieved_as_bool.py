import unittest
from traits.api import Bool, Dict, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
@requires_numpy
def test_numpy_bool_retrieved_as_bool(self):
    a = A()
    a.foo = numpy.bool_(True)
    self.assertIsInstance(a.foo, bool)
    a.foo = numpy.bool_(False)
    self.assertIsInstance(a.foo, bool)