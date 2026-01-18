import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import Constraint, ConcreteModel, TransformationFactory, Var
Test for violated trivial constraint upper bound.