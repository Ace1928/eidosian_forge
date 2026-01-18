import numpy as np
import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
Returns the solution to the original problem given the inverse_data.
        