import logging
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Var, inequality
from pyomo.util.infeasible import (
Test for logging of infeasible constraints.