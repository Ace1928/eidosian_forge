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