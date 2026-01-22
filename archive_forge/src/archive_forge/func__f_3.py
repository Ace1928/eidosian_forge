from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.ntheory import nextprime
from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.domains import ZZ
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly, PurePoly
from sympy.polys.polyutils import _analyze_gens
from sympy.utilities import subsets, public, filldedent
from sympy.polys.rings import ring
def _f_3():
    R, x, y, z = ring('x,y,z', ZZ)
    return x ** 5 * y ** 2 + x ** 4 * z ** 4 + x ** 4 + x ** 3 * y ** 3 * z + x ** 3 * z + x ** 2 * y ** 4 + x ** 2 * y ** 3 * z ** 3 + x ** 2 * y * z ** 5 + x ** 2 * y * z + x * y ** 2 * z ** 4 + x * y ** 2 + x * y * z ** 7 + x * y * z ** 3 + x * y * z ** 2 + y ** 2 * z + y * z ** 4