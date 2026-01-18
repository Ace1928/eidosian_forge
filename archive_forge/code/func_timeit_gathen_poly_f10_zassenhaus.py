from sympy.polys.galoistools import gf_from_dict, gf_factor_sqf
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
def timeit_gathen_poly_f10_zassenhaus():
    gf_factor_sqf(f_10, p_10, ZZ, method='zassenhaus')