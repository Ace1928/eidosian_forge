import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
class ScalarTester(ParamTester):

    def setUp(self, **kwds):
        self.model.Z = Set(initialize=[1, 3])
        self.model.A = Param(**kwds)
        self.instance = self.model.create_instance()
        self.expectTextDomainError = False
        self.expectNegativeDomainError = False

    def test_value_scalar(self):
        if self.data.get(None, NoValue) is NoValue:
            self.assertRaises(ValueError, value, self.instance.A)
            self.assertRaises(TypeError, float, self.instance.A)
            self.assertRaises(TypeError, int, self.instance.A)
        else:
            val = self.data[None]
            tmp = value(self.instance.A)
            self.assertEqual(type(tmp), type(val))
            self.assertEqual(tmp, val)
            self.assertRaises(TypeError, float, self.instance.A)
            self.assertRaises(TypeError, int, self.instance.A)

    def test_call(self):
        if self.sparse_data.get(None, 0) is NoValue or self.data.get(None, NoValue) is NoValue:
            self.assertRaisesRegex(ValueError, '.*currently set to an invalid value', self.instance.A.__call__)
        else:
            self.assertEqual(self.instance.A(), self.data[None])

    def test_get_valueattr(self):
        self.assertEqual(self.instance.A._value, self.sparse_data.get(None, NoValue))
        if self.data.get(None, 0) is NoValue:
            try:
                value(self.instance.A)
                self.fail('Expected value error')
            except ValueError:
                pass
        else:
            self.assertEqual(self.instance.A.value, self.data[None])

    def test_set_valueattr(self):
        self.instance.A.value = 4.3
        self.assertEqual(self.instance.A.value, 4.3)
        self.assertEqual(self.instance.A(), 4.3)

    def test_get_value(self):
        if self.sparse_data.get(None, 0) is NoValue or self.data.get(None, NoValue) is NoValue:
            try:
                value(self.instance.A)
                self.fail('Expected value error')
            except ValueError:
                pass
        else:
            self.assertEqual(self.instance.A.value, self.data[None])

    def test_set_value(self):
        self.instance.A = 4.3
        self.assertEqual(self.instance.A.value, 4.3)
        self.assertEqual(self.instance.A(), 4.3)

    def test_is_indexed(self):
        self.assertFalse(self.instance.A.is_indexed())

    def test_dim(self):
        self.assertEqual(self.instance.A.dim(), 0)