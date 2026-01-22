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
class IterationRangesRoot(IterationRanges):

    def __init__(self, name: str, numel: sympy.Expr, prefix: str, index: int, kernel: TritonKernel, pid_cache=None):
        if pid_cache is None:
            pid_cache = {}
        super().__init__(name=name, var_list=[], var_ranges={}, numel=numel, prefix=prefix, kernel=kernel)
        self.index = index
        self.nodes: Dict[sympy.Expr, IterationRangesEntry] = {}
        self.pid_cache: Dict[str, str] = pid_cache

    def cache_clear(self):
        for node in self.nodes.values():
            node.cache_clear()

    def lookup(self, divisor, length):
        """
        Lookup a given RangeTreeEntry, creating it if needed
        """
        if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
            expr = FloorDiv(sympy_symbol(f'{self.prefix}index'), divisor)
        else:
            expr = ModularIndexing(sympy_symbol(f'{self.prefix}index'), divisor, length)
        if expr not in self.nodes:
            node = IterationRangesEntry(f'{self.prefix}{next(V.kernel.iter_vars_count)}', divisor, length, expr, self)
            V.kernel.range_tree_nodes[node.symbol()] = node
            self.var_list.append(node.symbol())
            self.var_ranges[node.symbol()] = length
            self.nodes[expr] = node
        return self.nodes[expr]

    def construct_entries(self, lengths: List[sympy.Expr]):
        divisor = sympy.Integer(1)
        itervars = []
        for length in reversed(lengths):
            itervars.append(self.lookup(divisor, length))
            divisor = divisor * length
        return list(reversed(itervars))

    def construct(self, lengths: List[sympy.Expr]):
        return [e.symbol() for e in self.construct_entries(lengths)]

    def vars_and_sizes(self, index: sympy.Expr):
        """Figure out vars from this tree used in index"""
        nodes = [V.kernel.range_tree_nodes.get(s) for s in index.free_symbols]
        nodes = [n for n in nodes if n and n.prefix == self.prefix]
        nodes.sort(key=lambda x: V.graph.sizevars.size_hint(x.divisor))
        divisor = sympy.Integer(1)
        index_vars = []
        sizes = []

        def add(node):
            nonlocal divisor
            index_vars.append(node.symbol())
            sizes.append(node.length)
            divisor = divisor * node.length
        for node in nodes:
            if not V.graph.sizevars.statically_known_equals(node.divisor, divisor):
                add(self.lookup(divisor, FloorDiv(node.divisor, divisor)))
                divisor = node.divisor
            add(node)
        if not V.graph.sizevars.statically_known_equals(self.numel, divisor):
            add(self.lookup(divisor, FloorDiv(self.numel, divisor)))
        return (list(reversed(index_vars)), list(reversed(sizes)))

    def ranges_code(self):
        size = self.kernel.indexing_size_str(self.index, self.prefix)
        index_dtype = self.kernel.index_dtype
        convert = f'.to({index_dtype})' if index_dtype != 'tl.int32' else ''
        return f'tl.arange(0, {self.prefix.upper()}BLOCK){size}{convert}'

    def scalar_code(self, value):
        index_dtype = self.kernel.index_dtype
        ndim = self.kernel.triton_tensor_ndim()
        size = [1] * ndim
        return f'tl.full({size}, {value}, {index_dtype})'

    def get_pid(self):
        key = f'tl.program_id({self.index})'
        pid = self.pid_cache.get(key, key)
        if self.kernel.index_dtype != 'tl.int32':
            return f'{pid}.to({self.kernel.index_dtype})'
        return pid

    def codegen_header(self, code, no_x_dim=False):
        x = self.prefix
        if self.is_loop():
            code.writeline(f'{self.name} = {x}offset + {x}base')
        elif x == 'r' and self.kernel.persistent_reduction:
            code.writeline(f'{self.name} = {self.ranges_code()}')
        else:
            if not no_x_dim:
                line = f'{x}offset + {self.ranges_code()}'
            else:
                line = self.scalar_code(f'{x}offset')
            code.writelines([f'{x}offset = {self.get_pid()} * {x.upper()}BLOCK', f'{self.name} = {line}'])
        code.writeline(f'{x}mask = {self.name} < {x}numel')