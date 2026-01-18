import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
@unittest.skipUnless(numpy_available, 'This test requires NumPy')
def test_numpy_basic_float_registration(self):
    self.assertIn(numpy.float_, native_numeric_types)
    self.assertNotIn(numpy.float_, native_integer_types)
    self.assertIn(numpy.float_, _native_boolean_types)
    self.assertIn(numpy.float_, native_types)