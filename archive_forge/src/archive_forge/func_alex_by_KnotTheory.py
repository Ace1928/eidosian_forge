import sys
from . import links, tangles
def alex_by_KnotTheory(L):
    p = mathematica.MyAlex(L.PD_code(True)).sage()
    i = next(iter((i for i, c in enumerate(p) if c != 0)))
    R = PolynomialRing(ZZ, 'a')
    return R(p[i:])