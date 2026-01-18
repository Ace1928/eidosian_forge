from sage.all import QQ, PolynomialRing, NumberField
from .giac_helper import giac
def toK(f):
    return K(T(S(f)))