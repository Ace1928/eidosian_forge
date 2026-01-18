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
def test_polyadic_predicate():

    class SexyPredicate(Predicate):
        pass
    try:
        Q.sexyprime = SexyPredicate()

        @Q.sexyprime.register(Integer, Integer)
        def _(int1, int2, assumptions):
            args = sorted([int1, int2])
            if not all((ask(Q.prime(a), assumptions) for a in args)):
                return False
            return args[1] - args[0] == 6

        @Q.sexyprime.register(Integer, Integer, Integer)
        def _(int1, int2, int3, assumptions):
            args = sorted([int1, int2, int3])
            if not all((ask(Q.prime(a), assumptions) for a in args)):
                return False
            return args[2] - args[1] == 6 and args[1] - args[0] == 6
        assert ask(Q.sexyprime(5, 11))
        assert ask(Q.sexyprime(7, 13, 19))
    finally:
        del Q.sexyprime