import pyomo.common.unittest as unittest
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
from io import StringIO
def test_nothing_to_do(self):
    m = ConcreteModel()
    m.p = LogicalConstraint()
    TransformationFactory('core.logical_to_linear').apply_to(m)
    self.assertIsNone(m.component('logic_to_linear'))
    self.assertFalse(m.p.active)