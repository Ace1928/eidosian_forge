import collections
import dataclasses
import itertools
import logging
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from .codegen.common import index_prevent_reordering
from .utils import get_dtype_size, sympy_str, sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
class MemoryDep(typing.NamedTuple):
    name: str
    index: sympy.Expr
    var_names: Tuple[sympy.Symbol, ...]
    size: Tuple[sympy.Expr, ...]

    def __repr__(self):
        return f'MemoryDep({self.name!r}, {self.index}, {self.ranges})'

    @property
    def ranges(self) -> Dict[sympy.Symbol, sympy.Expr]:
        """{c0: 128, c1: 512, ...}"""
        return dict(zip(self.var_names, self.size))

    def get_numel(self) -> sympy.Expr:
        if self.is_indirect():
            numel = V.graph.get_numel(self.name)
        else:
            vars = set(self.index.free_symbols)
            numel = sympy.Integer(1)
            for var, size in zip(self.var_names, self.size):
                if var in vars:
                    numel = numel * size
        return numel

    def rename(self, renames: Dict[str, str]) -> 'MemoryDep':
        if self.name in renames:
            return MemoryDep(renames[self.name], self.index, var_names=self.var_names, size=self.size)
        return self

    def numbytes_hint(self):
        return V.graph.sizevars.size_hint(self.get_numel()) * get_dtype_size(V.graph.get_dtype(self.name))

    def has_unbacked_symbols(self):
        return len(free_unbacked_symbols(self.get_numel())) > 0

    def is_contiguous(self) -> bool:
        return isinstance(self.index, sympy.Symbol) and self.index in self.var_names

    def is_scalar(self) -> bool:
        if isinstance(self.index, sympy.Symbol):
            return self.index not in self.var_names and (not self.is_indirect())
        return isinstance(self.index, (int, sympy.Integer))

    def is_indirect(self) -> bool:
        return any((is_indirect(v.name) for v in self.index.free_symbols))