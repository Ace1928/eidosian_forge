from sympy.core.mod import Mod
from sympy.core.numbers import (I, oo, pi)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, sin)
from sympy.simplify.simplify import simplify
from sympy.core import Symbol, S, Rational, Integer, Dummy, Wild, Pow
from sympy.core.assumptions import (assumptions, check_assumptions,
from sympy.core.facts import InconsistentAssumptions
from sympy.core.random import seed
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.testing.pytest import raises, XFAIL
def test_other_symbol():
    x = Symbol('x', integer=True)
    assert x.is_integer is True
    assert x.is_real is True
    assert x.is_finite is True
    x = Symbol('x', integer=True, nonnegative=True)
    assert x.is_integer is True
    assert x.is_nonnegative is True
    assert x.is_negative is False
    assert x.is_positive is None
    assert x.is_finite is True
    x = Symbol('x', integer=True, nonpositive=True)
    assert x.is_integer is True
    assert x.is_nonpositive is True
    assert x.is_positive is False
    assert x.is_negative is None
    assert x.is_finite is True
    x = Symbol('x', odd=True)
    assert x.is_odd is True
    assert x.is_even is False
    assert x.is_integer is True
    assert x.is_finite is True
    x = Symbol('x', odd=False)
    assert x.is_odd is False
    assert x.is_even is None
    assert x.is_integer is None
    assert x.is_finite is None
    x = Symbol('x', even=True)
    assert x.is_even is True
    assert x.is_odd is False
    assert x.is_integer is True
    assert x.is_finite is True
    x = Symbol('x', even=False)
    assert x.is_even is False
    assert x.is_odd is None
    assert x.is_integer is None
    assert x.is_finite is None
    x = Symbol('x', integer=True, nonnegative=True)
    assert x.is_integer is True
    assert x.is_nonnegative is True
    assert x.is_finite is True
    x = Symbol('x', integer=True, nonpositive=True)
    assert x.is_integer is True
    assert x.is_nonpositive is True
    assert x.is_finite is True
    x = Symbol('x', rational=True)
    assert x.is_real is True
    assert x.is_finite is True
    x = Symbol('x', rational=False)
    assert x.is_real is None
    assert x.is_finite is None
    x = Symbol('x', irrational=True)
    assert x.is_real is True
    assert x.is_finite is True
    x = Symbol('x', irrational=False)
    assert x.is_real is None
    assert x.is_finite is None
    with raises(AttributeError):
        x.is_real = False
    x = Symbol('x', algebraic=True)
    assert x.is_transcendental is False
    x = Symbol('x', transcendental=True)
    assert x.is_algebraic is False
    assert x.is_rational is False
    assert x.is_integer is False