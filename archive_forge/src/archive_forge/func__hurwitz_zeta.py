import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
def _hurwitz_zeta(x, q, tolerance):
    """The Hurwitz zeta function, or the Riemann zeta function of two arguments.

    ``x`` must be greater than one and ``q`` must be positive.

    This function repeatedly computes subsequent partial sums until
    convergence, as decided by ``tolerance``.
    """
    z = 0
    z_prev = -float('inf')
    k = 0
    while abs(z - z_prev) > tolerance:
        z_prev = z
        z += 1 / (k + q) ** x
        k += 1
    return z