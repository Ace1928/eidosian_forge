from sympy.core import Function, S, sympify, NumberKind
from sympy.utilities.iterables import sift
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.operations import LatticeOp, ShortCircuit
from sympy.core.function import (Application, Lambda,
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.relational import Eq, Relational
from sympy.core.singleton import Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.rules import Transform
from sympy.core.logic import fuzzy_and, fuzzy_or, _torf
from sympy.core.traversal import walk
from sympy.core.numbers import Integer
from sympy.logic.boolalg import And, Or
@classmethod
def _new_args_filter(cls, arg_sequence):
    """
        Generator filtering args.

        first standard filter, for cls.zero and cls.identity.
        Also reshape ``Max(a, Max(b, c))`` to ``Max(a, b, c)``,
        and check arguments for comparability
        """
    for arg in arg_sequence:
        if not isinstance(arg, Expr) or arg.is_extended_real is False or (arg.is_number and (not arg.is_comparable)):
            raise ValueError("The argument '%s' is not comparable." % arg)
        if arg == cls.zero:
            raise ShortCircuit(arg)
        elif arg == cls.identity:
            continue
        elif arg.func == cls:
            yield from arg.args
        else:
            yield arg