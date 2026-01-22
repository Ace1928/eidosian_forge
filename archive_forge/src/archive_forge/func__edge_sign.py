from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
import sage.graphs.graph as graph
from sage.rings.rational_field import QQ
def _edge_sign(K, edge):
    """Returns the sign (+/- 1) associated to given edge in the black graph."""
    crossing = edge[2]
    if set(((crossing, 0), (crossing, 1))).issubset(set(edge[0])) or set(((crossing, 0), (crossing, 1))).issubset(set(edge[1])):
        return +1
    return -1