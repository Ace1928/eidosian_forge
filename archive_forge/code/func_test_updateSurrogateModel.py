import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
def test_updateSurrogateModel(self):
    self.interface.replaceExternalFunctionsWithVariables()
    self.interface.createConstraints()
    self.interface.model.x[0] = 2.0
    self.interface.model.z.set_values({0: 5.0, 1: 2.5, 2: -1.0})
    self.interface.data.basis_model_output[:] = 0
    self.interface.data.grad_basis_model_output[...] = 0
    self.interface.data.truth_model_output[:] = 0
    self.interface.data.grad_truth_model_output[...] = 0
    self.interface.data.value_of_ef_inputs[...] = 0
    self.interface.updateSurrogateModel()
    for key, val in self.interface.data.basis_model_output.items():
        self.assertEqual(value(val), 0)
    for key, val in self.interface.data.grad_basis_model_output.items():
        self.assertEqual(value(val), 0)
    for key, val in self.interface.data.truth_model_output.items():
        self.assertEqual(value(val), 0.8414709848078965)
    truth_grads = []
    for key, val in self.interface.data.grad_truth_model_output.items():
        truth_grads.append(value(val))
    self.assertEqual(truth_grads, [cos(1), -cos(1)])
    for key, val in self.interface.data.value_of_ef_inputs.items():
        self.assertEqual(value(self.interface.model.x[key[1]]), value(val))
    self.interface.model.x.set_values({0: 0, 1: 0})
    self.interface.updateSurrogateModel()
    for key, val in self.interface.data.basis_model_output.items():
        self.assertEqual(value(val), 0)
    for key, val in self.interface.data.grad_basis_model_output.items():
        self.assertEqual(value(val), 0)
    for key, val in self.interface.data.truth_model_output.items():
        self.assertEqual(value(val), 0)
    truth_grads = []
    for key, val in self.interface.data.grad_truth_model_output.items():
        truth_grads.append(value(val))
    self.assertEqual(truth_grads, [cos(0), -cos(0)])
    for key, val in self.interface.data.value_of_ef_inputs.items():
        self.assertEqual(value(self.interface.model.x[key[1]]), value(val))