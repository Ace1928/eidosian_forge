from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import atan
from sympy.polys.polyroots import roots
from sympy.polys.polytools import cancel
from sympy.polys.rootoftools import RootSum
from sympy.polys import Poly, resultant, ZZ
def _include_sign(c, sqf):
    if c.is_extended_real and (c < 0) == True:
        h, k = sqf[0]
        c_poly = c.as_poly(h.gens)
        sqf[0] = (h * c_poly, k)