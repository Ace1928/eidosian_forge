import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def arrows_around_edges(manifold):
    T = t3m.Mcomplex(manifold)
    starts = lex_first_edge_starts(T)
    ans = []
    return [faces_around_edge(T, tet, edge) for tet, edge in starts]