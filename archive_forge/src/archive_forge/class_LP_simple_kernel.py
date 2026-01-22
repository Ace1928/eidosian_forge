import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_simple_kernel(LP_simple):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.a1 = pmo.parameter(value=1.0)
        model.a2 = pmo.parameter_dict({1: pmo.parameter(value=1.0)})
        model.a3 = pmo.parameter(value=1.0)
        model.a4 = pmo.parameter_dict({1: pmo.parameter(value=1.0)})
        model.x = pmo.variable(domain=NonNegativeReals)
        model.y = pmo.variable(domain=NonNegativeReals)
        model.z1 = pmo.variable()
        model.z2 = pmo.variable()
        model.dummy_expr1 = pmo.expression(model.a1 * model.a2[1])
        model.dummy_expr2 = pmo.expression(model.y / model.a3 * model.a4[1])
        model.inactive_obj = pmo.objective(model.x + 3.0 * model.y + 1.0 + model.z1 - model.z2)
        model.inactive_obj.deactivate()
        model.p = pmo.parameter(value=0.0)
        model.obj = pmo.objective(model.p + model.inactive_obj)
        model.c1 = pmo.constraint(model.dummy_expr1 <= pmo.noclone(model.dummy_expr2))
        model.c2 = pmo.constraint((2.0, model.x / model.a3 - model.y, 10))
        model.c3 = pmo.constraint((0, model.z1 + 1, 10))
        model.c4 = pmo.constraint((-10, model.z2 + 1, 0))