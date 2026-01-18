import pyomo.common.unittest as unittest
from pyomo.contrib.gdp_bounds.info import disjunctive_lb, disjunctive_ub
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import check_available_solvers
Test computation of disjunctive bounds.