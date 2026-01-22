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
class ExternKernelSchedulerNode(BaseSchedulerNode):

    def debug_str_extra(self) -> str:
        return f'{self.get_name()}.node.kernel = {getattr(self.node, 'kernel', None)}'

    def is_extern(self):
        return True

    def has_side_effects(self):
        return hasattr(self.node, 'has_side_effects') and self.node.has_side_effects()

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if self.get_aliases() or self.is_template():
            return False
        if read_dep.name not in self.scheduler.name_to_node:
            return False
        if not isinstance(self.node, (torch._inductor.ir.AllReduce, torch._inductor.ir.InPlaceHint)):
            return False
        if len(self.read_writes.writes) == 1:
            write_dep = next(iter(self.read_writes.writes))
            numel_diff = read_dep.get_numel() - write_dep.get_numel()
            return V.graph.sizevars.simplify(numel_diff) == 0
        return False