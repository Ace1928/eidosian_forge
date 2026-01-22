from __future__ import annotations
from typing import TYPE_CHECKING
from sympy.simplify import simplify as simp, trigsimp as tsimp  # type: ignore
from sympy.core.decorators import call_highest_priority, _sympifyit
from sympy.core.assumptions import StdFactKB
from sympy.core.function import diff as df
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor as fctr
from sympy.core import S, Add, Mul
from sympy.core.expr import Expr
class BasisDependentZero(BasisDependent):
    """
    Class to denote a zero basis dependent instance.
    """
    components: dict['BaseVector', Expr] = {}
    _latex_form: str

    def __new__(cls):
        obj = super().__new__(cls)
        obj._hash = (S.Zero, cls).__hash__()
        return obj

    def __hash__(self):
        return self._hash

    @call_highest_priority('__req__')
    def __eq__(self, other):
        return isinstance(other, self._zero_func)
    __req__ = __eq__

    @call_highest_priority('__radd__')
    def __add__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError('Invalid argument types for addition')

    @call_highest_priority('__add__')
    def __radd__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError('Invalid argument types for addition')

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        if isinstance(other, self._expr_type):
            return -other
        else:
            raise TypeError('Invalid argument types for subtraction')

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError('Invalid argument types for subtraction')

    def __neg__(self):
        return self

    def normalize(self):
        """
        Returns the normalized version of this vector.
        """
        return self

    def _sympystr(self, printer):
        return '0'