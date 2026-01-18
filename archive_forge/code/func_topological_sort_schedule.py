import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
def topological_sort_schedule(self):
    """
        Ensure self.nodes is in topologically sorted order
        """
    seen: Set[ir.Buffer] = set()
    name_to_node: Dict[str, ir.Buffer] = dict()
    result: List[ir.Buffer] = []

    def visit(n):
        if n not in seen:
            seen.add(n)
            for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                visit(name_to_node[dep.name])
            result.append(n)
    for node in self.nodes:
        for name in node.get_names():
            name_to_node[name] = node
    for node in self.nodes:
        visit(node)
    self.nodes = result