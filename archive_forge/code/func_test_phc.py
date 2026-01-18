import snappy
from sage.all import QQ, PolynomialRing, matrix, prod
import giac_rur
from closed import zhs_exs
import phc_wrapper
def test_phc(manifold):
    G = manifold.fundamental_group(True, True, False)
    I = character_variety(G)
    return phc_wrapper.find_solutions(I)