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
def test_symbol_extended_real_false():
    a = Symbol('a', extended_real=False)
    assert a.is_real is False
    assert a.is_integer is False
    assert a.is_zero is False
    assert a.is_negative is False
    assert a.is_positive is False
    assert a.is_nonnegative is False
    assert a.is_nonpositive is False
    assert a.is_nonzero is False
    assert a.is_extended_negative is False
    assert a.is_extended_positive is False
    assert a.is_extended_nonnegative is False
    assert a.is_extended_nonpositive is False
    assert a.is_extended_nonzero is False