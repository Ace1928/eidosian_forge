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
def visit_modular_indexing(base, divisor, modulus):
    base = remove_zero_terms(base, divisor)
    base_pos = True
    if isinstance(base, ModularIndexing):
        base_s = base.args[2] - 1
    elif not base.has(ModularIndexing):
        iter_ranges_zero = {k: 0 for k, v in var_ranges.items()}
        base_lowest = sympy_subs(base, iter_ranges_zero)
        if self.statically_known_leq(0, base_lowest):
            base_pos = True
        else:
            base_pos = False
        iter_ranges = {k: v - 1 for k, v in var_ranges.items()}
        base_s = sympy_subs(base, iter_ranges)
    else:
        base_s = base
    if self.statically_known_lt(base_s, modulus * divisor) and base_pos:
        return FloorDiv(base, divisor)
    return ModularIndexing(base, divisor, modulus)