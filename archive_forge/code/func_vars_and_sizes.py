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