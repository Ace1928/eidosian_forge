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
class TritonPrinter(PythonPrinter):

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return f'tl.math.floor({self.paren(self._print(expr.args[0]))})'

    def _helper_sqrt(self, expr):
        return f'tl.math.sqrt({self.paren(self._print(expr))}.to(tl.float32))'

    def _print_Where(self, expr):
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f'tl.where({c}, {p}, {q})'

    def _print_Min(self, expr):
        nargs = len(expr.args)
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        mid = len(expr.args) // 2
        a = self._print(sympy.Min(*expr.args[:mid]))
        b = self._print(sympy.Min(*expr.args[mid:]))
        return f'tl.math.min({a}, {b})'

    def _print_Max(self, expr):
        nargs = len(expr.args)
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        mid = len(expr.args) // 2
        a = self._print(sympy.Max(*expr.args[:mid]))
        b = self._print(sympy.Max(*expr.args[mid:]))
        return f'tl.math.max({a}, {b})'

    def _print_Abs(self, expr):
        assert len(expr.args) == 1
        return f'tl.abs({self._print(expr.args[0])})'