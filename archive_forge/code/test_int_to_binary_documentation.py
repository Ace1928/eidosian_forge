import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Integers, value
from pyomo.environ import TransformationFactory as xfrm
from pyomo.common.log import LoggingIntercept
import logging
from io import StringIO
Tests integer to binary variable reformulation.