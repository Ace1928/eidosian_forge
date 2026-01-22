import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
from pyomo.repn.beta.matrix import compile_block_linear_constraints
def _generate_base_model(self):
    self.model = pmo.block()
    model = self.model
    model._name = self.description
    model.s = list(range(1, 13))
    model.x = pmo.variable_dict(((i, pmo.variable()) for i in model.s))
    model.x[1].lb = -1
    model.x[1].ub = 1
    model.x[2].lb = -1
    model.x[2].ub = 1
    model.obj = pmo.objective(expr=sum((model.x[i] * (-1) ** (i + 1) for i in model.s)))
    variable_order = [model.x[3], model.x[4], model.x[5], model.x[6], model.x[7], model.x[8], model.x[9], model.x[10], model.x[11], model.x[12]]
    return variable_order