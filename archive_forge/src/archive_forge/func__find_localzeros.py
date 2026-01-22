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
def _find_localzeros(cls, values, **options):
    """
        Sequentially allocate values to localzeros.

        When a value is identified as being more extreme than another member it
        replaces that member; if this is never true, then the value is simply
        appended to the localzeros.
        """
    localzeros = set()
    for v in values:
        is_newzero = True
        localzeros_ = list(localzeros)
        for z in localzeros_:
            if id(v) == id(z):
                is_newzero = False
            else:
                con = cls._is_connected(v, z)
                if con:
                    is_newzero = False
                    if con is True or con == cls:
                        localzeros.remove(z)
                        localzeros.update([v])
        if is_newzero:
            localzeros.update([v])
    return localzeros