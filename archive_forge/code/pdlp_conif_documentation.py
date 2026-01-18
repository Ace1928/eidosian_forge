import logging
from typing import Any, Dict, Tuple
import numpy as np
from scipy.sparse import csr_matrix
import cvxpy.settings as s
from cvxpy import Zero
from cvxpy.constraints import NonNeg
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version
Returns the result of the call to the solver.