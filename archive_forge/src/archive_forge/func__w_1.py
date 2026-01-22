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
def _w_1():
    R, x, y, z = ring('x,y,z', ZZ)
    return 4 * x ** 6 * y ** 4 * z ** 2 + 4 * x ** 6 * y ** 3 * z ** 3 - 4 * x ** 6 * y ** 2 * z ** 4 - 4 * x ** 6 * y * z ** 5 + x ** 5 * y ** 4 * z ** 3 + 12 * x ** 5 * y ** 3 * z - x ** 5 * y ** 2 * z ** 5 + 12 * x ** 5 * y ** 2 * z ** 2 - 12 * x ** 5 * y * z ** 3 - 12 * x ** 5 * z ** 4 + 8 * x ** 4 * y ** 4 + 6 * x ** 4 * y ** 3 * z ** 2 + 8 * x ** 4 * y ** 3 * z - 4 * x ** 4 * y ** 2 * z ** 4 + 4 * x ** 4 * y ** 2 * z ** 3 - 8 * x ** 4 * y ** 2 * z ** 2 - 4 * x ** 4 * y * z ** 5 - 2 * x ** 4 * y * z ** 4 - 8 * x ** 4 * y * z ** 3 + 2 * x ** 3 * y ** 4 * z + x ** 3 * y ** 3 * z ** 3 - x ** 3 * y ** 2 * z ** 5 - 2 * x ** 3 * y ** 2 * z ** 3 + 9 * x ** 3 * y ** 2 * z - 12 * x ** 3 * y * z ** 3 + 12 * x ** 3 * y * z ** 2 - 12 * x ** 3 * z ** 4 + 3 * x ** 3 * z ** 3 + 6 * x ** 2 * y ** 3 - 6 * x ** 2 * y ** 2 * z ** 2 + 8 * x ** 2 * y ** 2 * z - 2 * x ** 2 * y * z ** 4 - 8 * x ** 2 * y * z ** 3 + 2 * x ** 2 * y * z ** 2 + 2 * x * y ** 3 * z - 2 * x * y ** 2 * z ** 3 - 3 * x * y * z + 3 * x * z ** 3 - 2 * y ** 2 + 2 * y * z ** 2