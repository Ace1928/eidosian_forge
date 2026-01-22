import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

    An infeasible LP
    