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
class StarDep(typing.NamedTuple):
    name: str

    @property
    def index(self):
        raise NotImplementedError('StarDep does not have an index')

    def get_numel(self) -> sympy.Expr:
        return V.graph.get_numel(self.name)

    def rename(self, renames: Dict[str, str]) -> 'StarDep':
        if self.name in renames:
            return StarDep(renames[self.name])
        return self

    def numbytes_hint(self):
        return V.graph.sizevars.size_hint(self.get_numel()) * get_dtype_size(V.graph.get_dtype(self.name))

    def has_unbacked_symbols(self):
        return len(free_unbacked_symbols(self.get_numel())) > 0

    def is_contiguous(self) -> bool:
        return False

    def is_scalar(self) -> bool:
        return False

    def is_indirect(self) -> bool:
        return False