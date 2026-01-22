import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
class Test_value(unittest.TestCase):

    def test_none(self):
        val = None
        self.assertEqual(val, value(val))

    def test_bool(self):
        val = False
        self.assertEqual(val, value(val))

    def test_float(self):
        val = 1.1
        self.assertEqual(val, value(val))

    def test_int(self):
        val = 1
        self.assertEqual(val, value(val))

    def test_long(self):
        val = int(10000000000.0)
        self.assertEqual(val, value(val))

    def test_nan(self):
        val = nan
        self.assertEqual(id(val), id(value(val)))

    def test_inf(self):
        val = inf
        self.assertEqual(id(val), id(value(val)))

    def test_string(self):
        val = 'foo'
        self.assertEqual(val, value(val))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertEqual(1.0, value(val))

    def test_const3(self):
        val = NumericConstant(nan)
        self.assertEqual(id(nan), id(value(val)))

    def test_const4(self):
        val = NumericConstant(inf)
        self.assertEqual(id(inf), id(value(val)))

    def test_param1(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=2)
        self.assertEqual(2, value(m.p))

    def test_param2(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        self.assertRaises(ValueError, value, m.p, exception=True)

    def test_param3(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        self.assertEqual(None, value(m.p, exception=False))

    def test_var1(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertRaises(ValueError, value, m.x, exception=True)

    def test_var2(self):
        m = ConcreteModel()
        m.x = Var()
        self.assertEqual(None, value(m.x, exception=False))

    def test_error1(self):

        class A(object):
            pass
        val = A()
        with self.assertRaisesRegex(TypeError, 'Cannot evaluate object with unknown type: A'):
            value(val)

    def test_unknownType(self):
        ref = MyBogusType(42)
        with self.assertRaisesRegex(TypeError, 'Cannot evaluate object with unknown type: MyBogusType'):
            value(ref)

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        val = value(ref)
        self.assertEqual(val.val, 42.0)
        self.assertIn(MyBogusNumericType, native_numeric_types)
        self.assertIn(MyBogusNumericType, native_types)
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)