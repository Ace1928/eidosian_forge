import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, Binary, maximize
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

    A mixed-integer model with a quadratic objective and quadratic constraints
    