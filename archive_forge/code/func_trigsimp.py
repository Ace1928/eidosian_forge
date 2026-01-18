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
def trigsimp(self, **opts):
    """
        Implements the SymPy trigsimp routine, for this quantity.

        trigsimp's documentation
        ========================

        """
    trig_components = [tsimp(v, **opts) * k for k, v in self.components.items()]
    return self._add_func(*trig_components)