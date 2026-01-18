import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
@requires_numpy
def test_rejects_numpy_types(self):
    numpy_values = [numpy.float16(25), numpy.float32(25), numpy.float64(25)]
    for numpy_value in numpy_values:
        self.model.percentage = 88
        with self.assertRaises(TraitError):
            self.model.percentage = numpy_value
        self.assertEqual(self.model.percentage, 88)