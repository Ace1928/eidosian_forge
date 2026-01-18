from sympy.abc import t, w, x, y, z, n, k, m, p, i
from sympy.assumptions import (ask, AssumptionsContext, Q, register_handler,
from sympy.assumptions.assume import assuming, global_assumptions, Predicate
from sympy.assumptions.cnf import CNF, Literal
from sympy.assumptions.facts import (single_fact_lookup,
from sympy.assumptions.handlers import AskHandler
from sympy.assumptions.ask_generated import (get_all_known_facts,
from sympy.core.add import Add
from sympy.core.numbers import (I, Integer, Rational, oo, zoo, pi)
from sympy.core.singleton import S
from sympy.core.power import Pow
from sympy.core.symbol import Str, symbols, Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
from sympy.logic.boolalg import Equivalent, Implies, Xor, And, to_cnf
from sympy.matrices import Matrix, SparseMatrix
from sympy.testing.pytest import (XFAIL, slow, raises, warns_deprecated_sympy,
import math
def test_real_pow():
    assert ask(Q.real(x ** 2), Q.real(x)) is True
    assert ask(Q.real(sqrt(x)), Q.negative(x)) is False
    assert ask(Q.real(x ** y), Q.real(x) & Q.integer(y)) is True
    assert ask(Q.real(x ** y), Q.real(x) & Q.real(y)) is None
    assert ask(Q.real(x ** y), Q.positive(x) & Q.real(y)) is True
    assert ask(Q.real(x ** y), Q.imaginary(x) & Q.imaginary(y)) is None
    assert ask(Q.real(x ** y), Q.imaginary(x) & Q.real(y)) is None
    assert ask(Q.real(x ** y), Q.real(x) & Q.imaginary(y)) is None
    assert ask(Q.real(x ** 0), Q.imaginary(x)) is True
    assert ask(Q.real(x ** y), Q.real(x) & Q.integer(y)) is True
    assert ask(Q.real(x ** y), Q.positive(x) & Q.real(y)) is True
    assert ask(Q.real(x ** y), Q.real(x) & Q.rational(y)) is None
    assert ask(Q.real(x ** y), Q.imaginary(x) & Q.integer(y)) is None
    assert ask(Q.real(x ** y), Q.imaginary(x) & Q.odd(y)) is False
    assert ask(Q.real(x ** y), Q.imaginary(x) & Q.even(y)) is True
    assert ask(Q.real(x ** (y / z)), Q.real(x) & Q.real(y / z) & Q.rational(y / z) & Q.even(z) & Q.positive(x)) is True
    assert ask(Q.real(x ** (y / z)), Q.real(x) & Q.rational(y / z) & Q.even(z) & Q.negative(x)) is False
    assert ask(Q.real(x ** (y / z)), Q.real(x) & Q.integer(y / z)) is True
    assert ask(Q.real(x ** (y / z)), Q.real(x) & Q.real(y / z) & Q.positive(x)) is True
    assert ask(Q.real(x ** (y / z)), Q.real(x) & Q.real(y / z) & Q.negative(x)) is False
    assert ask(Q.real((-I) ** i), Q.imaginary(i)) is True
    assert ask(Q.real(I ** i), Q.imaginary(i)) is True
    assert ask(Q.real(i ** i), Q.imaginary(i)) is None
    assert ask(Q.real(x ** i), Q.imaginary(i)) is None
    assert ask(Q.real(x ** (I * pi / log(x))), Q.real(x)) is True