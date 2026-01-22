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
def _is_connected(cls, x, y):
    """
        Check if x and y are connected somehow.
        """
    for i in range(2):
        if x == y:
            return True
        t, f = (Max, Min)
        for op in '><':
            for j in range(2):
                try:
                    if op == '>':
                        v = x >= y
                    else:
                        v = x <= y
                except TypeError:
                    return False
                if not v.is_Relational:
                    return t if v else f
                t, f = (f, t)
                x, y = (y, x)
            x, y = (y, x)
        x = factor_terms(x - y)
        y = S.Zero
    return False