import os
import shutil
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.core.base.external import (
from pyomo.core.base.units_container import pint_available, units
from pyomo.core.expr.numeric_expr import (
from pyomo.opt import check_available_solvers
def test_partial_clone(self):
    m = ConcreteModel()
    m.f = ExternalFunction(_sum)
    m.x = Var(initialize=3)
    m.y = Var(initialize=5)
    m.b = Block()
    m.b.e = Expression(expr=m.f(m.x, m.y))
    self.assertIsInstance(m.f, PythonCallbackFunction)
    self.assertEqual(m.f._fcn_id, m.b.e.arg(0).arg(-1).value)
    self.assertEqual(value(m.b.e), 10)
    m.c = m.b.clone()
    self.assertIsNot(m.b.e, m.c.e)
    self.assertIsNot(m.b.e.arg(0), m.c.e.arg(0))
    self.assertEqual(m.f._fcn_id, m.b.e.arg(0).arg(-1).value)
    self.assertEqual(m.f._fcn_id, m.c.e.arg(0).arg(-1).value)
    self.assertEqual(value(m.c.e), 10)
    _fcn_id = m.f._fcn_id
    m.f.__setstate__(m.f.__getstate__())
    self.assertEqual(m.f._fcn_id, _fcn_id)
    self.assertIsNot(m.b.e, m.c.e)
    self.assertIsNot(m.b.e.arg(0), m.c.e.arg(0))
    self.assertEqual(m.f._fcn_id, m.b.e.arg(0).arg(-1).value)
    self.assertEqual(m.f._fcn_id, m.c.e.arg(0).arg(-1).value)
    self.assertEqual(value(m.c.e), 10)