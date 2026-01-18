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
def test_scalar_set_mutable_when_not_present(self):
    m = ConcreteModel()
    m.p = Param(mutable=True)
    self.assertEqual(m.p._data, {})
    m.p = 10
    self.assertEqual(len(m.p._data), 1)
    self.assertIs(m.p._data[None], m.p)
    m.x_p = Var(bounds=(0, m.p))
    self.assertEqual(m.x_p.bounds, (0, 10))
    m.p = 20
    self.assertEqual(m.x_p.bounds, (0, 20))