from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import (piecewise_fold,
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrices import MatrixBase
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.exceptions import sympy_deprecation_warning
def _eval_factor(self, **hints):
    if 1 == len(self.limits):
        summand = self.function.factor(**hints)
        if summand.is_Mul:
            out = sift(summand.args, lambda w: w.is_commutative and (not set(self.variables) & w.free_symbols))
            return Mul(*out[True]) * self.func(Mul(*out[False]), *self.limits)
    else:
        summand = self.func(self.function, *self.limits[0:-1]).factor()
        if not summand.has(self.variables[-1]):
            return self.func(1, [self.limits[-1]]).doit() * summand
        elif isinstance(summand, Mul):
            return self.func(summand, self.limits[-1]).factor()
    return self