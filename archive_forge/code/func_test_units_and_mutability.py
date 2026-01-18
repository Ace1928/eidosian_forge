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
@unittest.skipUnless(pint_available, 'units test requires pint module')
def test_units_and_mutability(self):
    m = ConcreteModel()
    with LoggingIntercept() as LOG:
        m.p = Param(units=units.g)
    self.assertEqual(LOG.getvalue(), '')
    self.assertTrue(m.p.mutable)
    with LoggingIntercept() as LOG:
        m.q = Param(units=units.g, mutable=True)
    self.assertEqual(LOG.getvalue(), '')
    self.assertTrue(m.q.mutable)
    with LoggingIntercept() as LOG:
        m.r = Param(units=units.g, mutable=False)
    self.assertEqual(LOG.getvalue(), "Params with units must be mutable.  Converting Param 'r' to mutable.\n")
    self.assertTrue(m.r.mutable)