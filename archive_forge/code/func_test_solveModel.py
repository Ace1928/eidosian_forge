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
def test_solveModel(self):
    self.interface.replaceExternalFunctionsWithVariables()
    self.interface.createConstraints()
    self.interface.data.basis_constraint.activate()
    self.interface.model.x[0] = 2.0
    self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
    self.interface.data.basis_model_output[:] = 0
    self.interface.data.grad_basis_model_output[...] = 0
    self.interface.data.truth_model_output[:] = 0
    self.interface.data.grad_truth_model_output[...] = 0
    self.interface.data.value_of_ef_inputs[...] = 0
    objective, step_norm, feasibility = self.interface.solveModel()
    self.assertAlmostEqual(objective, 5.150744273013601)
    self.assertAlmostEqual(step_norm, 3.393437471478297)
    self.assertAlmostEqual(feasibility, 0.09569982275514467)
    self.interface.data.basis_constraint.deactivate()
    self.interface.updateSurrogateModel()
    self.interface.data.sm_constraint_basis.activate()
    objective, step_norm, feasibility = self.interface.solveModel()
    self.assertAlmostEqual(objective, 5.15065981284333)
    self.assertAlmostEqual(step_norm, 0.0017225116628372117)
    self.assertAlmostEqual(feasibility, 0.00014665023773349772)