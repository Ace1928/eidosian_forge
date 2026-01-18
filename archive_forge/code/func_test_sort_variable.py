from sympy.concrete.summations import Sum
from sympy.core.basic import Basic, _aresame
from sympy.core.cache import clear_cache
from sympy.core.containers import Dict, Tuple
from sympy.core.expr import Expr, unchanged
from sympy.core.function import (Subs, Function, diff, Lambda, expand,
from sympy.core.numbers import E, Float, zoo, Rational, pi, I, oo, nan
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Dummy, Symbol
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, acos
from sympy.functions.special.error_functions import expint
from sympy.functions.special.gamma_functions import loggamma, polygamma
from sympy.matrices.dense import Matrix
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.tensor.indexed import Indexed
from sympy.core.function import (PoleError, _mexpand, arity,
from sympy.core.parameters import _exp_is_pow
from sympy.core.sympify import sympify, SympifyError
from sympy.matrices import MutableMatrix, ImmutableMatrix
from sympy.sets.sets import FiniteSet
from sympy.solvers.solveset import solveset
from sympy.tensor.array import NDimArray
from sympy.utilities.iterables import subsets, variations
from sympy.testing.pytest import XFAIL, raises, warns_deprecated_sympy, _both_exp_pow
from sympy.abc import t, w, x, y, z
def test_sort_variable():
    vsort = Derivative._sort_variable_count

    def vsort0(*v, reverse=False):
        return [i[0] for i in vsort([(i, 0) for i in (reversed(v) if reverse else v)])]
    for R in range(2):
        assert vsort0(y, x, reverse=R) == [x, y]
        assert vsort0(f(x), x, reverse=R) == [x, f(x)]
        assert vsort0(f(y), f(x), reverse=R) == [f(x), f(y)]
        assert vsort0(g(x), f(y), reverse=R) == [f(y), g(x)]
        assert vsort0(f(x, y), f(x), reverse=R) == [f(x), f(x, y)]
        fx = f(x).diff(x)
        assert vsort0(fx, y, reverse=R) == [y, fx]
        fy = f(y).diff(y)
        assert vsort0(fy, fx, reverse=R) == [fx, fy]
        fxx = fx.diff(x)
        assert vsort0(fxx, fx, reverse=R) == [fx, fxx]
        assert vsort0(Basic(x), f(x), reverse=R) == [f(x), Basic(x)]
        assert vsort0(Basic(y), Basic(x), reverse=R) == [Basic(x), Basic(y)]
        assert vsort0(Basic(y, z), Basic(x), reverse=R) == [Basic(x), Basic(y, z)]
        assert vsort0(fx, x, reverse=R) == [x, fx] if R else [fx, x]
        assert vsort0(Basic(x), x, reverse=R) == [x, Basic(x)] if R else [Basic(x), x]
        assert vsort0(Basic(f(x)), f(x), reverse=R) == [f(x), Basic(f(x))] if R else [Basic(f(x)), f(x)]
        assert vsort0(Basic(x, z), Basic(x), reverse=R) == [Basic(x), Basic(x, z)] if R else [Basic(x, z), Basic(x)]
    assert vsort([]) == []
    assert _aresame(vsort([(x, 1)]), [Tuple(x, 1)])
    assert vsort([(x, y), (x, z)]) == [(x, y + z)]
    assert vsort([(y, 1), (x, 1 + y)]) == [(x, 1 + y), (y, 1)]
    assert vsort([(x, 3), (y, 2), (z, 1)]) == [(x, 3), (y, 2), (z, 1)]
    assert vsort([(h(x), 1), (g(x), 1), (f(x), 1)]) == [(f(x), 1), (g(x), 1), (h(x), 1)]
    assert vsort([(z, 1), (y, 2), (x, 3), (h(x), 1), (g(x), 1), (f(x), 1)]) == [(x, 3), (y, 2), (z, 1), (f(x), 1), (g(x), 1), (h(x), 1)]
    assert vsort([(x, 1), (f(x), 1), (y, 1), (f(y), 1)]) == [(x, 1), (y, 1), (f(x), 1), (f(y), 1)]
    assert vsort([(y, 1), (x, 2), (g(x), 1), (f(x), 1), (z, 1), (h(x), 1), (y, 2), (x, 1)]) == [(x, 3), (y, 3), (z, 1), (f(x), 1), (g(x), 1), (h(x), 1)]
    assert vsort([(z, 1), (y, 1), (f(x), 1), (x, 1), (f(x), 1), (g(x), 1)]) == [(x, 1), (y, 1), (z, 1), (f(x), 2), (g(x), 1)]
    assert vsort([(z, 1), (y, 2), (f(x), 1), (x, 2), (f(x), 2), (g(x), 1), (z, 2), (z, 1), (y, 1), (x, 1)]) == [(x, 3), (y, 3), (z, 4), (f(x), 3), (g(x), 1)]
    assert vsort(((y, 2), (x, 1), (y, 1), (x, 1))) == [(x, 2), (y, 3)]
    assert isinstance(vsort([(x, 3), (y, 2), (z, 1)])[0], Tuple)
    assert vsort([(x, 1), (f(x), 1), (x, 1)]) == [(x, 2), (f(x), 1)]
    assert vsort([(y, 2), (x, 3), (z, 1)]) == [(x, 3), (y, 2), (z, 1)]
    assert vsort([(h(y), 1), (g(x), 1), (f(x), 1)]) == [(f(x), 1), (g(x), 1), (h(y), 1)]
    assert vsort([(x, 1), (y, 1), (x, 1)]) == [(x, 2), (y, 1)]
    assert vsort([(f(x), 1), (f(y), 1), (f(x), 1)]) == [(f(x), 2), (f(y), 1)]
    dfx = f(x).diff(x)
    self = [(dfx, 1), (x, 1)]
    assert vsort(self) == self
    assert vsort([(dfx, 1), (y, 1), (f(x), 1), (x, 1), (f(y), 1), (x, 1)]) == [(y, 1), (f(x), 1), (f(y), 1), (dfx, 1), (x, 2)]
    dfy = f(y).diff(y)
    assert vsort([(dfy, 1), (dfx, 1)]) == [(dfx, 1), (dfy, 1)]
    d2fx = dfx.diff(x)
    assert vsort([(d2fx, 1), (dfx, 1)]) == [(dfx, 1), (d2fx, 1)]