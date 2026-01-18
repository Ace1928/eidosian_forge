from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def update_tr_radius(Delta, actual_reduction, predicted_reduction, step_norm, bound_hit):
    """Update the radius of a trust region based on the cost reduction.

    Returns
    -------
    Delta : float
        New radius.
    ratio : float
        Ratio between actual and predicted reductions.
    """
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1
    else:
        ratio = 0
    if ratio < 0.25:
        Delta = 0.25 * step_norm
    elif ratio > 0.75 and bound_hit:
        Delta *= 2.0
    return (Delta, ratio)