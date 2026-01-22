import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, Binary
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class MILP_infeasible1_kernel(MILP_infeasible1):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description
        model.x = pmo.variable(domain=Binary)
        model.y = pmo.variable(domain=Binary)
        model.z = pmo.variable(domain=Binary)
        model.o = pmo.objective(-model.x - model.y - model.z)
        model.c1 = pmo.constraint(model.x + model.y <= 1)
        model.c2 = pmo.constraint(model.x + model.z <= 1)
        model.c3 = pmo.constraint(model.y + model.z <= 1)
        model.c4 = pmo.constraint(model.x + model.y + model.z >= 1.5)