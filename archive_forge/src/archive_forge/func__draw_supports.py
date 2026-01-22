from sympy.core import S, Symbol, diff, symbols
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.sympify import sympify
from sympy.solvers import linsolve
from sympy.solvers.ode.ode import dsolve
from sympy.solvers.solvers import solve
from sympy.printing import sstr
from sympy.functions import SingularityFunction, Piecewise, factorial
from sympy.integrals import integrate
from sympy.series import limit
from sympy.plotting import plot, PlotGrid
from sympy.geometry.entity import GeometryEntity
from sympy.external import import_module
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import iterable
def _draw_supports(self, length, l):
    height = float(length / 10)
    support_markers = []
    support_rectangles = []
    for support in self._applied_supports:
        if l:
            pos = support[0].subs(l)
        else:
            pos = support[0]
        if support[1] == 'pin':
            support_markers.append({'args': [pos, [0]], 'marker': 6, 'markersize': 13, 'color': 'black'})
        elif support[1] == 'roller':
            support_markers.append({'args': [pos, [-height / 2.5]], 'marker': 'o', 'markersize': 11, 'color': 'black'})
        elif support[1] == 'fixed':
            if pos == 0:
                support_rectangles.append({'xy': (0, -3 * height), 'width': -length / 20, 'height': 6 * height + height, 'fill': False, 'hatch': '/////'})
            else:
                support_rectangles.append({'xy': (length, -3 * height), 'width': length / 20, 'height': 6 * height + height, 'fill': False, 'hatch': '/////'})
    return (support_markers, support_rectangles)