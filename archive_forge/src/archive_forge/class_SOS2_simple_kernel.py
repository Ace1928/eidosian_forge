import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class SOS2_simple_kernel(SOS2_simple):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.f = pmo.variable()
        model.x = pmo.variable(lb=1, ub=3)
        model.fi = pmo.parameter_dict()
        model.fi[1] = pmo.parameter(value=1.0)
        model.fi[2] = pmo.parameter(value=2.0)
        model.fi[3] = pmo.parameter(value=0.0)
        model.xi = pmo.parameter_dict()
        model.xi[1] = pmo.parameter(value=1.0)
        model.xi[2] = pmo.parameter(value=2.0)
        model.xi[3] = pmo.parameter(value=3.0)
        model.p = pmo.variable(domain=NonNegativeReals)
        model.n = pmo.variable(domain=NonNegativeReals)
        model.lmbda = pmo.variable_dict(((i, pmo.variable()) for i in range(1, 4)))
        model.obj = pmo.objective(model.p + model.n)
        model.c1 = pmo.constraint_dict()
        model.c1[1] = pmo.constraint((0.0, model.lmbda[1], 1.0))
        model.c1[2] = pmo.constraint((0.0, model.lmbda[2], 1.0))
        model.c1[3] = pmo.constraint(0.0 <= model.lmbda[3])
        model.c2 = pmo.sos2(model.lmbda.values())
        model.c3 = pmo.constraint(sum(model.lmbda.values()) == 1)
        model.c4 = pmo.constraint(model.f == sum((model.fi[i] * model.lmbda[i] for i in model.lmbda)))
        model.c5 = pmo.constraint(model.x == sum((model.xi[i] * model.lmbda[i] for i in model.lmbda)))
        model.x.fix(2.75)
        model.c6 = pmo.sos2([])