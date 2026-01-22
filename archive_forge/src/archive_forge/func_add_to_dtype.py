import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
def add_to_dtype(sub_graph: torch.fx.Graph):

    def is_lowp_fp_load(node: torch.fx.Node):
        if node.target not in ['load']:
            return False
        assert len(node.args) == 3
        load_dtype = V.graph.get_dtype(node.args[1])
        return load_dtype in DTYPE_LOWP_FP

    def is_lowp_fp_store(node: torch.fx.Node):
        if node.target != 'store':
            return False
        _, store_var, _, _, _ = node.args
        store_dtype = V.graph.get_dtype(store_var)
        return store_dtype in DTYPE_LOWP_FP
    sub_graph_nodes = list(sub_graph.nodes)
    to_lowp_fp_legalized_nodes = []
    for _node in sub_graph_nodes:
        if is_lowp_fp_load(_node):
            if all((user.target == 'store' for user in _node.users)):
                continue
            ops = _node.args[0]
            with sub_graph.inserting_after(_node):
                to_type_node = sub_graph.call_method('to_dtype', args=(ops, _node, torch.float))
                to_type_node_args = to_type_node.args
                _node.replace_all_uses_with(to_type_node)
                to_type_node.args = to_type_node_args
                metrics.cpp_to_dtype_count += 1
        elif is_lowp_fp_store(_node):
            ops, name, _, value_var, _ = _node.args
            if value_var.target == 'load' and all((user.target == 'store' for user in value_var.users)):
                continue
            dtype = V.graph.get_dtype(name)
            with sub_graph.inserting_before(_node):
                to_type_node = sub_graph.call_method('to_dtype', args=(ops, value_var, dtype))
                _node.replace_input_with(value_var, to_type_node)
                metrics.cpp_to_dtype_count += 1
        elif _node.target == 'reduction':
            ops, dtype, src_dtype, reduction_type, value = _node.args
            if src_dtype in DTYPE_LOWP_FP:
                assert dtype in [torch.float, torch.bfloat16, torch.float16, torch.int64]
                _node.args = (ops, torch.float if dtype in DTYPE_LOWP_FP else dtype, torch.float, reduction_type, value)
        elif _node.target == 'to_dtype' and _node.args[-1] in DTYPE_LOWP_FP:
            ops, x, _ = _node.args
            to_lowp_fp_legalized_nodes.append(_node)
            _node.args = (ops, x, torch.float)
        else:
            pass

    def eliminate_to_dtype(sub_graph: torch.fx.Graph):

        def _eliminate_duplicate_to_node(sub_graph: torch.fx.Graph):

            def _used_by_to(to_node: torch.fx.Node):
                return all((usr.target == 'to_dtype' for usr in to_node.users))
            all_to_nodes = [node for node in sub_graph.nodes if node.target == 'to_dtype']
            all_to_nodes_and_users = [{node: node.users} for node in all_to_nodes if _used_by_to(node)]
            for node_users in all_to_nodes_and_users:
                for node, users in node_users.items():
                    if node in sub_graph.nodes and (all((usr.args[-1] == node.args[-1] for usr in users)) or (node in to_lowp_fp_legalized_nodes and all((usr.args[-1] in DTYPE_LOWP_FP for usr in users)))):
                        val_node = node.all_input_nodes[-1]
                        node.replace_all_uses_with(val_node)
                        sub_graph.erase_node(node)
            if sub_graph.owning_module is None:
                sub_graph.lint()
        _eliminate_duplicate_to_node(sub_graph)
    eliminate_to_dtype(sub_graph)