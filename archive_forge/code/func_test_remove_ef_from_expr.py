import logging
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.environ import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.visitor import identify_variables
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.TRF import _trf_config
def test_remove_ef_from_expr(self):
    self.interface.data.all_variables = ComponentSet()
    self.interface.data.truth_models = ComponentMap()
    self.interface.data.ef_outputs = VarList()
    self.interface.data.basis_expressions = ComponentMap()
    component = self.interface.model.obj
    self.interface._remove_ef_from_expr(component)
    self.assertEqual(str(self.interface.model.obj.expr), '(z[0] - 1.0)**2 + (z[0] - z[1])**2 + (z[2] - 1.0)**2 + (x[0] - 1.0)**4 + (x[1] - 1.0)**6')
    component = self.interface.model.c1
    str_expr = str(component.expr)
    self.interface._remove_ef_from_expr(component)
    self.assertNotEqual(str_expr, str(component.expr))
    self.assertEqual(str(component.expr), 'x[0]*z[0]**2 + trf_data.ef_outputs[1]  ==  2.8284271247461903')