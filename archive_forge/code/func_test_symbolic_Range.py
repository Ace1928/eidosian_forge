from sympy.core.expr import unchanged
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range, normalize_theta_set,
from sympy.sets.sets import (FiniteSet, Interval, Union, imageset,
from sympy.sets.conditionset import ConditionSet
from sympy.simplify.simplify import simplify
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import And
from sympy.matrices.dense import eye
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, y, t, z
from sympy.core.mod import Mod
import itertools
def test_symbolic_Range():
    n = Symbol('n')
    raises(ValueError, lambda: Range(n)[0])
    raises(IndexError, lambda: Range(n, n)[0])
    raises(ValueError, lambda: Range(n, n + 1)[0])
    raises(ValueError, lambda: Range(n).size)
    n = Symbol('n', integer=True)
    raises(ValueError, lambda: Range(n)[0])
    raises(IndexError, lambda: Range(n, n)[0])
    assert Range(n, n + 1)[0] == n
    raises(ValueError, lambda: Range(n).size)
    assert Range(n, n + 1).size == 1
    n = Symbol('n', integer=True, nonnegative=True)
    raises(ValueError, lambda: Range(n)[0])
    raises(IndexError, lambda: Range(n, n)[0])
    assert Range(n + 1)[0] == 0
    assert Range(n, n + 1)[0] == n
    assert Range(n).size == n
    assert Range(n + 1).size == n + 1
    assert Range(n, n + 1).size == 1
    n = Symbol('n', integer=True, positive=True)
    assert Range(n)[0] == 0
    assert Range(n, n + 1)[0] == n
    assert Range(n).size == n
    assert Range(n, n + 1).size == 1
    m = Symbol('m', integer=True, positive=True)
    assert Range(n, n + m)[0] == n
    assert Range(n, n + m).size == m
    assert Range(n, n + 1).size == 1
    assert Range(n, n + m, 2).size == floor(m / 2)
    m = Symbol('m', integer=True, positive=True, even=True)
    assert Range(n, n + m, 2).size == m / 2