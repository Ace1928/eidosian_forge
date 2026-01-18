import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
def test_replaceExternalFunctionsWithVariables(self):
    self.interface.replaceExternalFunctionsWithVariables()
    for var in self.interface.model.component_data_objects(Var):
        self.assertIn(var, ComponentSet(self.interface.data.all_variables))
    for i in self.interface.data.ef_outputs:
        self.assertIn(self.interface.data.ef_outputs[i], ComponentSet(self.interface.data.all_variables))
    for i, k in self.interface.data.truth_models.items():
        self.assertIsInstance(k, ExternalFunctionExpression)
        self.assertIn(str(self.interface.model.x[0]), str(k))
        self.assertIn(str(self.interface.model.x[1]), str(k))
        self.assertIsInstance(i, _GeneralVarData)
        self.assertEqual(i, self.interface.data.ef_outputs[1])
    for i, k in self.interface.data.basis_expressions.items():
        self.assertEqual(k, 0)
        self.assertEqual(i, self.interface.data.ef_outputs[1])
    self.assertEqual(1, list(self.interface.data.ef_inputs.keys())[0])
    self.assertEqual(self.interface.data.ef_inputs[1], [self.interface.model.x[0], self.interface.model.x[1]])
    self.assertEqual(list(self.interface.model.component_objects(ExternalFunction)), [])
    self.m.obj2 = Objective(expr=self.m.x[0] ** 2 - (self.m.z[1] - 3) ** 3)
    interface = TRFInterface(self.m, [self.m.z[0], self.m.z[1], self.m.z[2]], self.ext_fcn_surrogate_map_rule, self.config)
    with self.assertRaises(ValueError):
        interface.replaceExternalFunctionsWithVariables()