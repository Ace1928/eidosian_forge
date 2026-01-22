import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import check_available_solvers
def _generateModel():
    model = ConcreteModel()
    model.x = Var(within=Binary)
    model.y = Var()
    model.c1 = Constraint(expr=model.y >= model.x)
    model.c2 = Constraint(expr=model.y >= 1.5 - model.x)
    model.obj = Objective(expr=model.y)
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    return model