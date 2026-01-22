from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence
def _find_x(func):
    free = func.free_symbols
    if len(free) == 1:
        return free.pop()
    elif not free:
        return Dummy('k')
    else:
        raise ValueError(' specify dummy variables for %s. If the function contains more than one free symbol, a dummy variable should be supplied explicitly e.g. FourierSeries(m*n**2, (n, -pi, pi))' % func)