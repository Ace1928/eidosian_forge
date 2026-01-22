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
class _TargetArgsExpr(_TargetExpr):
    """
    Base class for filtering match by node.{target,args,kwargs}
    """

    def __init__(self, fns, *args, _users=1, **kwargs):
        super().__init__(fns, _users)
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        if any((isinstance(x, (dict, list, tuple)) for x in itertools.chain(args, kwargs.values()))):
            self.flatten = self.pytree_flatten
        else:
            self.flatten = self.simple_flatten
        self.flat_args_kwargs = self.flatten(self.args, self.kwargs)

    @staticmethod
    def simple_flatten(args, kwargs: Dict[Any, Any]):
        return ((*args, *kwargs.values()), (len(args), *kwargs.keys()))

    @staticmethod
    def pytree_flatten(args, kwargs: Dict[Any, Any]):

        def norm_spec(s: pytree.TreeSpec):
            if s.type is None:
                return s
            mapping = {immutable_list: list, tuple: list, immutable_dict: dict}
            return pytree.TreeSpec(mapping.get(s.type, s.type), s.context, list(map(norm_spec, s.children_specs)))
        flat, spec = pytree.tree_flatten([args, kwargs])
        spec = norm_spec(spec)
        return (flat, spec)

    def __repr__(self):
        args = [self.fns_repr(), *map(repr, self.args), *[f'{k}={v}' for k, v in self.kwargs.items()]]
        return f'{self.__class__.__name__}({', '.join(args)})'

    def pretty_print(self, pp: PatternPrettyPrinter):
        args = [self.fns_repr(), *(pp.pretty_print(x) for x in self.args), *[f'{k}={pp.pretty_print(v)}' for k, v in self.kwargs.items()]]
        if isinstance(self.users, Multiple):
            args.append('_users=MULTIPLE')
        elif self.users > 1:
            args.append(f'_users={self.users}')
        joiner_str = ', '
        return f'{self.__class__.__name__}({joiner_str.join(args)})'

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        if not self._match_fns(node) or len(node.args) != len(self.args):
            return FailedMatch('function_mismatch: node={}, pattern={}', node, self)
        if not self._match_users(node, ctx):
            return FailedMatch('multiple_users {}', self)
        _args = node.args
        _kwargs = node.kwargs
        if len(_kwargs) < len(self.kwargs):
            from torch.fx.operator_schemas import normalize_function
            normalized_args_and_kwargs = normalize_function(node.target, node.args, node.kwargs)
            if normalized_args_and_kwargs is None:
                return FailedMatch('function_mismatch: node={}, pattern={}', node, self)
            else:
                _args, _kwargs = normalized_args_and_kwargs
                if len(_args) == len(self.args) and len(_kwargs) >= len(self.kwargs):
                    _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}
                else:
                    return FailedMatch('function_mismatch: node={}, pattern={}', node, self)
        else:
            _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}
        node_items, node_spec = self.flatten(_args, _kwargs)
        self_items, self_spec = self.flat_args_kwargs
        if node_spec != self_spec:
            return FailedMatch('args_structure {} {}', node_spec, self_spec)
        assert len(node_items) == len(self_items)
        m = Match(self)
        for i, pattern, child_node in zip(itertools.count(), self_items, node_items):
            if isinstance(pattern, PatternExpr):
                child_match = ctx.match(pattern, child_node)
                if not child_match:
                    return child_match
                m.extend(child_match)
            elif isinstance(child_node, torch.fx.Node) or child_node != pattern:
                return FailedMatch('constant_args: {} {!r}!={pattern!r}', node, child_node)
        m.nodes.append(node)
        m.targets[self] = node.target
        return m

    def find_anchor_nodes(self, ctx: MatchContext, searched):
        """
        This is used when we are matching a pattern with multiple outputs.
        There is a partial match (stored in ctx) and we want to walk
        this pattern to find a connection to an already-matched node.

        Yields candidate nodes that `self._match` might like.
        """
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]
            return
        for pattern in self.flat_args_kwargs[0]:
            if isinstance(pattern, PatternExpr):
                for other_node in pattern.find_anchor_nodes(ctx, searched):
                    if not isinstance(other_node, torch.fx.Node):
                        continue
                    for node in other_node.users:
                        if node not in searched:
                            if self._match_fns(node):
                                yield node
                                searched.add(node)