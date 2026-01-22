from typing import Type
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import ImmutableMatrix, eye
from sympy.matrices.expressions import MatMul, MatAdd
from sympy.polys import Poly, rootof
from sympy.polys.polyroots import roots
from sympy.polys.polytools import (cancel, degree)
from sympy.series import limit
from mpmath.libmp.libmpf import prec_to_dps
def _eval_subs(self, old, new):
    arg_num = self.num.subs(old, new)
    arg_den = self.den.subs(old, new)
    argnew = TransferFunction(arg_num, arg_den, self.var)
    return self if old == self.var else argnew