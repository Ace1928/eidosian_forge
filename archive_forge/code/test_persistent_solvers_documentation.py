import pyomo.environ as pe
from pyomo.common.dependencies import attempt_import
import pyomo.common.unittest as unittest
from pyomo.contrib.appsi.base import TerminationCondition, Results, PersistentSolver
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.appsi.solvers import Gurobi, Ipopt, Cplex, Cbc, Highs
from typing import Type
from pyomo.core.expr.numeric_expr import LinearExpression
import os
import random
from pyomo import gdp

        This test is for a bug where an objective containing a fixed variable does
        not get updated properly when the variable is unfixed.
        