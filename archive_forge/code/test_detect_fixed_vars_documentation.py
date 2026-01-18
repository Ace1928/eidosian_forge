import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, TransformationFactory, Var, value
Test for reversion of fixed variables.