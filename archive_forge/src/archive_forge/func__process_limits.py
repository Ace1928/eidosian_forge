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
def _process_limits(func, limits):
    """
    Limits should be of the form (x, start, stop).
    x should be a symbol. Both start and stop should be bounded.

    Explanation
    ===========

    * If x is not given, x is determined from func.
    * If limits is None. Limit of the form (x, -pi, pi) is returned.

    Examples
    ========

    >>> from sympy.series.fourier import _process_limits as pari
    >>> from sympy.abc import x
    >>> pari(x**2, (x, -2, 2))
    (x, -2, 2)
    >>> pari(x**2, (-2, 2))
    (x, -2, 2)
    >>> pari(x**2, None)
    (x, -pi, pi)
    """

    def _find_x(func):
        free = func.free_symbols
        if len(free) == 1:
            return free.pop()
        elif not free:
            return Dummy('k')
        else:
            raise ValueError(' specify dummy variables for %s. If the function contains more than one free symbol, a dummy variable should be supplied explicitly e.g. FourierSeries(m*n**2, (n, -pi, pi))' % func)
    x, start, stop = (None, None, None)
    if limits is None:
        x, start, stop = (_find_x(func), -pi, pi)
    if is_sequence(limits, Tuple):
        if len(limits) == 3:
            x, start, stop = limits
        elif len(limits) == 2:
            x = _find_x(func)
            start, stop = limits
    if not isinstance(x, Symbol) or start is None or stop is None:
        raise ValueError('Invalid limits given: %s' % str(limits))
    unbounded = [S.NegativeInfinity, S.Infinity]
    if start in unbounded or stop in unbounded:
        raise ValueError('Both the start and end value should be bounded')
    return sympify((x, start, stop))