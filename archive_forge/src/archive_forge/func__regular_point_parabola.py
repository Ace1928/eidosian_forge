from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import gcd
from sympy.sets.sets import Complement
from sympy.core import Basic, Tuple, diff, expand, Eq, Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol
from sympy.solvers import solveset, nonlinsolve, diophantine
from sympy.polys import total_degree
from sympy.geometry import Point
from sympy.ntheory.factor_ import core
def _regular_point_parabola(self, a, b, c, d, e, f):
    ok = (a, d) != (0, 0) and (c, e) != (0, 0) and (b ** 2 == 4 * a * c) and ((a, c) != (0, 0))
    if not ok:
        raise ValueError('Rational Point on the conic does not exist')
    if a != 0:
        d_dash, f_dash = (4 * a * e - 2 * b * d, 4 * a * f - d ** 2)
        if d_dash != 0:
            y_reg = -f_dash / d_dash
            x_reg = -(d + b * y_reg) / (2 * a)
        else:
            ok = False
    elif c != 0:
        d_dash, f_dash = (4 * c * d - 2 * b * e, 4 * c * f - e ** 2)
        if d_dash != 0:
            x_reg = -f_dash / d_dash
            y_reg = -(e + b * x_reg) / (2 * c)
        else:
            ok = False
    if ok:
        return (x_reg, y_reg)
    else:
        raise ValueError('Rational Point on the conic does not exist')