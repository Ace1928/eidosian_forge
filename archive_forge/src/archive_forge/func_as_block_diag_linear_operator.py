from typing import Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, Zero
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.solver import Solver
def as_block_diag_linear_operator(matrices) -> LinearOperator:
    """Block diag of SciPy sparse matrices or linear operators."""
    linear_operators = [as_linear_operator(op) for op in matrices]
    nrows = [op.shape[0] for op in linear_operators]
    ncols = [op.shape[1] for op in linear_operators]
    m, n = (sum(nrows), sum(ncols))
    col_indices = np.append(0, np.cumsum(ncols))

    def matmul(X):
        outputs = []
        for i, op in enumerate(linear_operators):
            Xi = X[col_indices[i]:col_indices[i + 1]]
            outputs.append(op(Xi))
        return sp.vstack(outputs)
    return LinearOperator(matmul, (m, n))