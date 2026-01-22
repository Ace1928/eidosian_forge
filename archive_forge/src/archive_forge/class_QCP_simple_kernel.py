import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class QCP_simple_kernel(QCP_simple):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.x = pmo.variable(domain=NonNegativeReals)
        model.y = pmo.variable(domain=NonNegativeReals)
        model.z = pmo.variable(domain=NonNegativeReals)
        model.fixed_var = pmo.variable()
        model.fixed_var.fix(0.2)
        model.q1 = pmo.variable(ub=0.2)
        model.q2 = pmo.variable(lb=-2)
        model.obj = pmo.objective(model.x + model.q1 - model.q2, sense=maximize)
        model.c0 = pmo.constraint(model.x + model.y + model.z == 1)
        model.qc0 = pmo.constraint(model.x ** 2 + model.y ** 2 + model.fixed_var <= model.z ** 2)
        model.qc1 = pmo.constraint(model.x ** 2 <= model.y * model.z)
        model.c = pmo.constraint_dict()
        model.c[1] = pmo.constraint(lb=0, body=-model.q1 ** 2 + model.fixed_var)
        model.c[2] = pmo.constraint(body=model.q2 ** 2 + model.fixed_var, ub=5)