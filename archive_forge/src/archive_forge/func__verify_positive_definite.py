import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
@staticmethod
def _verify_positive_definite(matrix):
    """
        Verify that a given symmetric square matrix is positive
        definite. An exception is raised if the square matrix
        is not positive definite.

        Parameters
        ----------
        matrix : (N, N) array_like
            Candidate matrix.

        Raises
        ------
        ValueError
            If matrix is not symmetric, not positive definite,
            or the square roots of the diagonal entries are
            not accessible.
        LinAlgError
            If matrix is not invertible.
        """
    matrix = np.array(matrix)
    if not np.allclose(matrix, matrix.T, atol=1e-08):
        raise ValueError('Shape matrix must be symmetric.')
    np.linalg.inv(matrix)
    eigvals = np.linalg.eigvals(matrix)
    if np.min(eigvals) < 0:
        raise ValueError(f'Non positive-definite shape matrix (detected eigenvalues {eigvals})')
    for diag_entry in np.diagonal(matrix):
        if np.isnan(np.power(diag_entry, 0.5)):
            raise ValueError(f'Cannot evaluate square root of the diagonal entry {diag_entry} of argument `shape_matrix`. Check that this entry is nonnegative')