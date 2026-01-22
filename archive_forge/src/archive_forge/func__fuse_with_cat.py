import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def _fuse_with_cat(gm: IterGraphModule, comm_blocks: List[CommBlock], node_indices: Dict[fx.Node, int]) -> CommBlock:
    """Fuse the CommBlocks using concat given a list of CommBlock (only allreduce)."""
    last_input_node = comm_blocks[0].inputs[0]
    last_input_index = -1
    all_input_nodes = []
    for comm_block in comm_blocks:
        input_node = comm_block.inputs[0]
        if input_node.name.startswith('clone'):
            input_node = cast(fx.Node, input_node.args[0])
        all_input_nodes.append(input_node)
        index = node_indices[input_node]
        if index >= last_input_index:
            assert index != last_input_index
            last_input_node = input_node
            last_input_index = index
    with gm.graph.inserting_after(last_input_node):
        cat_inputs = []
        for input_node in all_input_nodes:
            cat_inputs.append(_call_function(gm, fake_tensor_mode, None, aten.flatten.using_ints, input_node))
    with gm.graph.inserting_after(cat_inputs[0]):
        cat_node = _call_function(gm, fake_tensor_mode, None, aten.cat, cat_inputs)
    last_comm = comm_blocks[-1]
    last_comm_node = last_comm.comm_node
    last_wait_node = last_comm.wait_nodes[0]
    with gm.graph.inserting_after(cat_node):
        flatten_args, spec = tree_flatten((last_comm_node.args, last_comm_node.kwargs))
        flatten_args[0] = cat_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_comm_node = _call_function(gm, fake_tensor_mode, cat_node.meta['val'], last_comm_node.target, *args, **kwargs)
    with gm.graph.inserting_after(fused_comm_node):
        flatten_args, spec = tree_flatten((last_wait_node.args, last_wait_node.kwargs))
        flatten_args[0] = fused_comm_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_wait_node = _call_function(gm, fake_tensor_mode, cat_node.meta['val'], last_wait_node.target, *args, **kwargs)
    nodes_to_move = cat_inputs + [cat_node, fused_comm_node, fused_wait_node]
    gm.graph.move_after(nodes_to_move, last_input_node)
    tensor_meta = cat_node.meta.get('tensor_meta')
    fused_comm_block = CommBlock(shape=tensor_meta.shape, node_list=[fused_comm_node, fused_wait_node], wait_nodes=[fused_wait_node], comm_node=fused_comm_node, inputs=[cat_node], outputs={fused_wait_node})
    _scatter_wait_result(gm, fused_comm_block, comm_blocks, node_indices)
    return fused_comm_block