from sympy.polys.galoistools import gf_from_dict, gf_factor_sqf
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
def timeit_shoup_poly_F10_zassenhaus():
    gf_factor_sqf(F_10, P_08, ZZ, method='zassenhaus')