import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
class Test_is_numeric_data(unittest.TestCase):

    def test_string(self):
        self.assertEqual(is_numeric_data('a'), False)
        self.assertEqual(is_numeric_data(b'a'), False)

    def test_float(self):
        self.assertEqual(is_numeric_data(0.0), True)

    def test_int(self):
        self.assertEqual(is_numeric_data(0), True)

    def test_NumericValue(self):
        self.assertEqual(is_numeric_data(NumericConstant(1.0)), True)

    def test_error(self):

        class A(object):
            pass
        val = A()
        self.assertEqual(False, is_numeric_data(val))

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertTrue(is_numeric_data(ref))
        self.assertIn(MyBogusNumericType, native_numeric_types)
        self.assertIn(MyBogusNumericType, native_types)
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)