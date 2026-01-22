import numpy as np
from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval
def _nc_hypergeom_mean_inverse(x, M, n, N):
    """
    For the given noncentral hypergeometric parameters x, M, n,and N
    (table[0,0], total, row 0 sum and column 0 sum, resp., of a 2x2
    contingency table), find the noncentrality parameter of Fisher's
    noncentral hypergeometric distribution whose mean is x.
    """
    nc = _solve(lambda nc: nchypergeom_fisher.mean(M, n, N, nc) - x)
    return nc