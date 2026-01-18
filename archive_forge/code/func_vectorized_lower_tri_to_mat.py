from __future__ import annotations
import warnings
from collections import defaultdict
import numpy as np
import scipy as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cone2cone.affine2direct import Dualize, Slacks
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.utilities import expcone_permutor
def vectorized_lower_tri_to_mat(v, dim):
    """
    :param v: a list of length (dim * (dim + 1) / 2)
    :param dim: the number of rows (equivalently, columns) in the output array.
    :return: Return the symmetric 2D array defined by taking "v" to
      specify its lower triangular entries.
    """
    rows, cols, vals = vectorized_lower_tri_to_triples(v, dim)
    A = sp.sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim)).toarray()
    d = np.diag(np.diag(A))
    A = A + A.T - d
    return A