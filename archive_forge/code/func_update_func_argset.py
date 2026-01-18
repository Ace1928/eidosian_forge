from collections import defaultdict
from sympy.core import Basic, Mul, Add, Pow, sympify
from sympy.core.containers import Tuple, OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
from sympy.matrices.expressions import (MatrixExpr, MatrixSymbol, MatMul,
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.polys.rootoftools import RootOf
from sympy.utilities.iterables import numbered_symbols, sift, \
from . import cse_opts
def update_func_argset(self, func_i, new_argset):
    """
        Update a function with a new set of arguments.
        """
    new_args = OrderedSet(new_argset)
    old_args = self.func_to_argset[func_i]
    for deleted_arg in old_args - new_args:
        self.arg_to_funcset[deleted_arg].remove(func_i)
    for added_arg in new_args - old_args:
        self.arg_to_funcset[added_arg].add(func_i)
    self.func_to_argset[func_i].clear()
    self.func_to_argset[func_i].update(new_args)