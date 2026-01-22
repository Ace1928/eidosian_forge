import string
from ..sage_helper import _within_sage, sage_method
def alexander_polynomial_basic(G, phi):
    R = phi.range()
    P = R.polynomial_ring()
    M = [[fox_derivative(rel, phi, var) for rel in G.relators()] for var in G.generators()]
    minexp = minimum_exponents(join_lists(M))
    M = matrix(P, [[convert_laurent_to_poly(p, minexp, P) for p in row] for row in M])
    alex_poly = gcd(M.minors(G.num_generators() - 1))
    if alex_poly == 0:
        return alex_poly
    return convert_laurent_to_poly(alex_poly, minimum_exponents([alex_poly]), P)