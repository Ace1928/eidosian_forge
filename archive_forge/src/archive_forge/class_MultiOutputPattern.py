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
class MultiOutputPattern(PatternExpr):

    def __init__(self, outputs):
        super().__init__()
        assert all((isinstance(x, (PatternExpr, type(None))) for x in outputs)), outputs
        self.outputs: List[Optional[PatternExpr]] = outputs

    @property
    def fns(self):
        assert self.outputs[0] and hasattr(self.outputs[0], 'fns')
        return self.outputs[0].fns

    def __repr__(self):
        return f'{self.__class__.__name__}({self.outputs})'

    def pretty_print(self, pp: PatternPrettyPrinter):
        args = [pp.pretty_print(x) for x in self.outputs]
        joiner_str = f',\n{'  '}'
        str_out = f'{self.__class__.__name__}([{joiner_str.join(args)}'
        str_out = f'{str_out}\n])'
        return str_out

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        m = ctx.match(self.outputs[0], node)
        if not m:
            return m
        for pattern in self.outputs[1:]:
            if pattern is None:
                continue
            child_match = self._match_from_anchors(pattern, ctx)
            if not child_match:
                return child_match
            m.extend(child_match)
        return m

    def _match_from_anchors(self, pattern, ctx):
        prior = dict(ctx.pattern_to_node)
        m = FailedMatch('no anchor found')
        for node in pattern.find_anchor_nodes(ctx, set()):
            m = ctx.match(pattern, node)
            if m:
                return m
            ctx.pattern_to_node = dict(prior)
        return m

    def match(self, node: torch.fx.Node) -> Union[Match, FailedMatch]:
        try:
            return MatchContext(self.outputs, graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e