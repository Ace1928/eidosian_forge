import snappy
from sage.all import QQ, PolynomialRing, matrix, prod
import giac_rur
from closed import zhs_exs
import phc_wrapper
def test_rur(manifold):
    G = manifold.fundamental_group(True, True, False)
    I = character_variety(G)
    return giac_rur.rational_univariate_representation(I)