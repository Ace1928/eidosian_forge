from collections.abc import Iterable
from functools import singledispatch
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.parameters import global_parameters
def _get_args_shapes(self):
    from sympy.tensor.array import Array
    return [i.shape if hasattr(i, 'shape') else Array(i).shape for i in self.args]