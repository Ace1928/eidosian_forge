import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import ExpCone
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import (
Returns solution to original problem, given inverse_data.
        