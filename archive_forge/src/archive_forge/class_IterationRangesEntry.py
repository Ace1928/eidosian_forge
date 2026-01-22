from __future__ import annotations
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
from typing import Any, Counter, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import MemoryDep, StarDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseScheduling, WhyNoFuse
from ..triton_heuristics import AutotuneHint
from ..utils import (
from ..virtualized import ops, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import (
from .triton_utils import config_of, signature_of, signature_to_meta
class IterationRangesEntry(IterationRanges):

    def __init__(self, name: str, divisor: sympy.Expr, length: sympy.Expr, expr: sympy.Expr, parent: IterationRanges):
        super().__init__(name=name, numel=parent.numel / length, var_list=parent.var_list, var_ranges=parent.var_ranges, prefix=parent.prefix, divisor=divisor, length=length, kernel=parent.kernel)
        self.parent = parent
        self.codegen = functools.lru_cache(None)(self._codegen)
        self.expr = expr

    def set_name(self, name):
        self.codegen = lambda: name
        self.codegen.cache_clear = lambda: None
        self.name = name

    def cache_clear(self):
        self.codegen.cache_clear()

    def writeline(self, line):
        if self.is_loop():
            V.kernel.indexing_code.writeline(line)
        else:
            V.kernel.body.writeline(line)

    def _codegen(self):
        self.writeline(f'{self.name} = ' + texpr(V.kernel.rename_indexing(self.expr)))
        return self.name

    def precomputed_args(self):
        precomputed_args: List[sympy.Expr] = []
        if isinstance(self.expr, sympy.Symbol):
            return precomputed_args
        assert isinstance(self.expr, (FloorDiv, ModularIndexing)), type(self.expr)
        for arg in self.expr.args[1:]:
            if not isinstance(arg, (sympy.Integer, sympy.Symbol)):
                symbols = arg.free_symbols
                if len(symbols) > 0 and all((s.name.startswith('s') for s in symbols)):
                    precomputed_args.append(arg)
        return precomputed_args

    def symbol(self):
        return sympy_symbol(self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name