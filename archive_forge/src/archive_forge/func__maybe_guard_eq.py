import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
@lru_cache(256)
def _maybe_guard_eq(self, expr: Union['sympy.Eq', 'sympy.Ne'], concrete_bool: bool) -> None:
    """
        Evaluates the result of an eq call. If true, uses information to
        simplify shapes (i.e. a == b or a % 5 == 0)
        """
    assert type(concrete_bool) is bool
    if isinstance(expr, sympy.Eq):
        if not concrete_bool:
            return
    elif isinstance(expr, sympy.Ne):
        if concrete_bool:
            return
    free = list(expr.free_symbols)
    assert len(free) > 0, f'The expression should not be static by this point: {expr}'
    if len(free) > 5:
        return
    free = sorted(free, key=lambda x: (self.size_hint(x, allow_none=True) or sys.maxsize, x.name), reverse=True)
    lhs = expr.lhs
    rhs = expr.rhs
    if not expr.has(Mod):
        try:
            floor_div_atoms = lhs.atoms(FloorDiv).union(rhs.atoms(FloorDiv))
            if len(floor_div_atoms) > 0 and any((a.divisor != 1 for a in floor_div_atoms)):
                raise NotImplementedError
            if isinstance(lhs, sympy.Symbol) and free_unbacked_symbols(lhs):
                self._set_replacement(lhs, self._find(rhs))
            elif isinstance(rhs, sympy.Symbol) and free_unbacked_symbols(rhs):
                self._set_replacement(rhs, self._find(lhs))
            else:
                r = try_solve(expr, free[0], floordiv_inequality=False)
                if r is not None and all((t.is_integer for t in sympy.preorder_traversal(r[1]))):
                    new_var = self._find(r[1])
                    ok = False
                    if self.is_unbacked_symint(free[0]):
                        ok = len(free_unbacked_symbols(new_var)) <= 1
                    else:
                        ok = len(free_unbacked_symbols(new_var)) == 0
                    if ok:
                        self._set_replacement(cast(sympy.Symbol, free[0]), new_var)
        except NotImplementedError:
            pass
    if expr.has(Mod):
        mod_expr = next(iter(expr.atoms(Mod)))
        try:
            r = try_solve(expr, mod_expr, floordiv_inequality=False)
            if r is not None and r[1] == 0:
                self._add_divisible(mod_expr)
                p, q = mod_expr.args
                if isinstance(q, sympy.Number) and isinstance(p, sympy.Mul) and (len(p.args) == 2):
                    c, i0 = p.args
                    if isinstance(c, sympy.Number) and isinstance(i0, sympy.Symbol) and self.is_unbacked_symint(i0):
                        d = q / sympy.gcd(q, c)
                        i1 = self.create_unbacked_symint().node.expr
                        self.var_to_range[i1] = SymPyValueRangeAnalysis.truediv(self.var_to_range[i0], ValueRanges.wrap(d))
                        self.runtime_var_to_range[i1] = SymPyValueRangeAnalysis.truediv(self.runtime_var_to_range[i0], ValueRanges.wrap(d))
                        self._set_replacement(i0, d * i1)
        except NotImplementedError:
            pass
    return