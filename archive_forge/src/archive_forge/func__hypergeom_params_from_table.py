import numpy as np
from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval
def _hypergeom_params_from_table(table):
    x = table[0, 0]
    M = table.sum()
    n = table[0].sum()
    N = table[:, 0].sum()
    return (x, M, n, N)