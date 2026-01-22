import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class SOS1_simple_kernel(SOS1_simple):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.a = pmo.parameter(value=0.1)
        model.x = pmo.variable(domain=NonNegativeReals)
        model.y = pmo.variable_dict()
        model.y[1] = pmo.variable(domain=NonNegativeReals)
        model.y[2] = pmo.variable(domain=NonNegativeReals)
        model.obj = pmo.objective(model.x + model.y[1] + 2 * model.y[2])
        model.c1 = pmo.constraint(model.a <= model.y[2])
        model.c2 = pmo.constraint((2.0, model.x, 10.0))
        model.c3 = pmo.sos1(model.y.values())
        model.c4 = pmo.constraint(sum(model.y.values()) == 1)
        model.c5 = pmo.sos1([])