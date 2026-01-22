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
class WhyNoFuse:
    __slots__ = ['node1', 'node2', 'reason', 'args']
    reason: str
    args: Tuple[Any, ...]

    def __init__(self, node1: 'BaseSchedulerNode', node2: 'BaseSchedulerNode'):
        self.node1 = node1
        self.node2 = node2

    def __call__(self, reason, *args):
        self.reason = reason
        self.args = args
        fusion_log.debug(self)

    def __str__(self):
        return f'cannot fuse {self.node1.get_name()} with {self.node2.get_name()}: ' + self.reason % self.args