import functools
import itertools
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import bound_sympy
from .utils import sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
def _stride_vars(self, index: Expr, vars: List[sympy.Symbol], support_vars: List[sympy.Symbol]) -> List[Expr]:
    """Convert an indexing expression back into strides

        NOTE: This is only valid if the index is a standard strided offset
        calculation. e.g. 10 * ModularIndexing(i0 + 1, 1, 2) would give a
        stride of -10 because the index wraps around after the first element

        """
    strides = []
    index = self.simplify(index)
    index = index - sympy_subs(index, {v: sympy.Integer(0) for v in support_vars if v != 0})
    for i in range(len(vars)):
        index_dim = sympy_subs(index, {support_vars[j]: sympy.Integer(0) for j in range(len(support_vars)) if vars[i] != support_vars[j] and support_vars[j] != 0})
        v = vars[i]
        if v == 0:
            strides.append(sympy.Integer(0))
        else:
            strides.append(sympy_subs(index_dim, {v: sympy.Integer(1)}) - sympy_subs(index_dim, {v: sympy.Integer(0)}))
    return strides