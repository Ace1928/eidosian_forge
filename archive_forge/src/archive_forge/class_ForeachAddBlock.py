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
@dataclass(unsafe_hash=True)
class ForeachAddBlock:
    add_node: fx.Node
    generate_output: bool
    outputs: List[fx.Node] = field(default_factory=list)

    def generate_outputs(self):
        graph = self.add_node.graph
        for i, arg in enumerate(cast(Tuple[Any, ...], self.add_node.args[0])):
            with graph.inserting_after(self.add_node):
                updated_arg = graph.call_function(operator.getitem, (self.add_node, i))
            with graph.inserting_after(updated_arg):
                output_copy = graph.call_function(aten.copy_, (arg, updated_arg))
            self.outputs.append(output_copy)
        assert self.outputs, f'The output for {self.add_node} is empty.'

    def populate_outputs(self):
        self.outputs = [self.add_node for _ in cast(Tuple[Any, ...], self.add_node.args[0])]
        for updated_arg in self.add_node.users:
            assert updated_arg.target == operator.getitem, f'Unexpected node target {updated_arg.target}'
            idx = cast(int, updated_arg.args[1])
            output_copy = next(iter(updated_arg.users))
            assert str(output_copy.target).startswith('aten.copy_'), f'The execpted output node is different, {str(output_copy.target)}'
            self.outputs[idx] = output_copy
        for i, output in enumerate(self.outputs):
            assert output != self.add_node, f'{i}th output is not replaced.'

    def __post_init__(self):
        if self.outputs:
            return
        if self.generate_output:
            self.generate_outputs()
        else:
            self.populate_outputs()