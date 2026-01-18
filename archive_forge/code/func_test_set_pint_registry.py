import math
import pickle
from pyomo.common.errors import PyomoException
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.util.check_units import assert_units_consistent, check_units_equivalent
from pyomo.core.expr import inequality
from pyomo.core.expr.numvalue import NumericConstant
import pyomo.core.expr as EXPR
from pyomo.core.base.units_container import (
from io import StringIO
def test_set_pint_registry(self):
    um = _DeferredUnitsSingleton()
    pint_reg = pint_module.UnitRegistry()
    with LoggingIntercept() as LOG:
        um.set_pint_registry(pint_reg)
    self.assertEqual(LOG.getvalue(), '')
    self.assertIs(um.pint_registry, pint_reg)
    with LoggingIntercept() as LOG:
        um.set_pint_registry(pint_reg)
    self.assertEqual(LOG.getvalue(), '')
    with LoggingIntercept() as LOG:
        um.set_pint_registry(pint_module.UnitRegistry())
    self.assertIn('Changing the pint registry used by the Pyomo Units system after the PyomoUnitsContainer was constructed', LOG.getvalue())