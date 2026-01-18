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
def with_read(self, dep: Dep) -> 'ReadWrites':
    assert isinstance(dep, (WeakDep, StarDep))
    return ReadWrites(set.union(self.reads, {dep}), self.writes, self.index_exprs, self.range_vars, self.var_ranges, op_counts=self.op_counts)