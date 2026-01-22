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
class _OpCounter:
    """Shim to count how many times each op is used"""

    def __init__(self, inner):
        super().__init__()
        self.parent_handler = inner
        self._op_counts: typing.Counter[Any] = collections.Counter()

    def __getattr__(self, name):
        self._op_counts[name] += 1
        return getattr(self.parent_handler, name)