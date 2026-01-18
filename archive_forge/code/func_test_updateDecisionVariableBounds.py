import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
@unittest.skipIf(not SolverFactory('ipopt').available(False), 'The IPOPT solver is not available')
def test_updateDecisionVariableBounds(self):
    self.interface.initializeProblem()
    for var in self.interface.decision_variables:
        self.assertEqual(self.interface.initial_decision_bounds[var.name], [var.lb, var.ub])
    self.interface.updateDecisionVariableBounds(0.5)
    for var in self.interface.decision_variables:
        self.assertNotEqual(self.interface.initial_decision_bounds[var.name], [var.lb, var.ub])