import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
class DataTypePropagation:

    def __init__(self, body) -> None:
        self.body = body
        self.graphs: Dict[Union[Callable[..., Any], str], Any] = {'root': body.root_block.graph}
        for k, v in body.subblocks.items():
            self.graphs[k] = v.graph

    def deduce_node_dtype_by_inputs(self, node: torch.fx.Node):
        inputs = node.all_input_nodes
        input_nodes = [n for n in inputs if isinstance(n, torch.fx.Node) and n.op != 'placeholder']
        if len(input_nodes) == 0:
            return None
        all_input_nodes_propogated = all((OptimizationContext.key in n.meta and n.meta[OptimizationContext.key].dtype is not None for n in input_nodes))
        if not all_input_nodes_propogated:
            return None
        return functools.reduce(torch.promote_types, [n.meta[OptimizationContext.key].dtype for n in input_nodes])

    def deduce_node_dtype_by_subgraph(self, node: torch.fx.Node):
        sub_graph = self.graphs[node.target]
        dtype = self.propagate_graph(sub_graph)
        assert dtype
        return dtype

    def deduce_node_dtype(self, node: torch.fx.Node):
        if node.target in boolean_ops():
            return torch.bool
        if node.op == 'placeholder':
            return None
        if node.target == 'output':
            if len(node.args) != 1:
                return None
        if node.target in ('to_dtype', 'index_expr'):
            return node.args[-1]
        if node.target in ('rand', 'randn'):
            return torch.float
        if node.target in ('get_index', 'index_expr'):
            return torch.int64
        if node.target in ('load', 'store', 'store_reduction'):
            buf_name = node.args[1]
            return V.graph.get_dtype(buf_name)
        if node.target == operator.getitem:
            return self.deduce_node_dtype(node.args[0])
        assert isinstance(node.target, str)
        if node.target == 'reduction':
            return node.args[1]
        if node.target == 'constant':
            return DTYPE_TO_COMPUTATION_DTYPE[node.args[-1]]
        if node.target.startswith('masked_subblock'):
            return self.deduce_node_dtype_by_subgraph(node)
        return self.deduce_node_dtype_by_inputs(node)

    def propagate_graph(self, graph: torch.fx.Graph):
        assert graph.nodes
        graph_dtype = None
        for node in graph.nodes:
            if OptimizationContext.key in node.meta:
                opt_ctx = node.meta[OptimizationContext.key]
            else:
                opt_ctx = OptimizationContext()
            opt_ctx.dtype = self.deduce_node_dtype(node)
            node.meta[OptimizationContext.key] = opt_ctx
            if node.target == 'output':
                graph_dtype = opt_ctx.dtype
        return graph_dtype

    def propagate(self):
        self.propagate_graph(self.graphs['root'])

    @classmethod
    def propagate_loopbody(cls, body):
        return cls(body).propagate()

    @classmethod
    def propagate_scheduler_node(cls, node):
        from ..ir import LoopBody
        from ..scheduler import SchedulerNode
        assert isinstance(node, SchedulerNode)
        assert isinstance(node._body, LoopBody)
        DataTypePropagation.propagate_loopbody(node._body)