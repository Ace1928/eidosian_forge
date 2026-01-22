from __future__ import annotations
import dataclasses
import functools
import inspect
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import (
from typing_extensions import TypeGuard
import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.immutable_collections import immutable_dict, immutable_list
from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type
class RepeatedExpr(PatternExpr):
    """
    Checks for a repeated pattern. Useful for repeated operations after a node such as `split` or `unbind`
    """

    def __init__(self, inner_pattern: PatternExpr):
        super().__init__()
        assert hasattr(inner_pattern, 'fns')
        self.inner_pattern = inner_pattern

    @property
    def fns(self):
        return self.inner_pattern.fns

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        m = ctx.match(self.inner_pattern, node)
        if not m:
            return m
        ctx.pattern_to_node.pop(self.inner_pattern)
        for anchor_node in self.inner_pattern.find_anchor_nodes(ctx, set()):
            anchor_m = MatchContext([self], graph=node.graph).match(self.inner_pattern, anchor_node)
            if not anchor_m:
                return anchor_m
            m.extend(anchor_m)
        return m