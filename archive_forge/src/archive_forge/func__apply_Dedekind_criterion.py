from sympy.polys.polytools import Poly
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.utilities.decorator import public
from .modules import ModuleEndomorphism, ModuleHomomorphism, PowerBasis
from .utilities import extract_fundamental_discriminant
def _apply_Dedekind_criterion(T, p):
    """
    Apply the "Dedekind criterion" to test whether the order needs to be
    enlarged relative to a given prime *p*.
    """
    x = T.gen
    T_bar = Poly(T, modulus=p)
    lc, fl = T_bar.factor_list()
    assert lc == 1
    g_bar = Poly(1, x, modulus=p)
    for ti_bar, _ in fl:
        g_bar *= ti_bar
    h_bar = T_bar // g_bar
    g = Poly(g_bar, domain=ZZ)
    h = Poly(h_bar, domain=ZZ)
    f = (g * h - T) // p
    f_bar = Poly(f, modulus=p)
    Z_bar = f_bar
    for b in [g_bar, h_bar]:
        Z_bar = Z_bar.gcd(b)
    U_bar = T_bar // Z_bar
    m = Z_bar.degree()
    return (U_bar, m)