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
class FusedAdamBlock:
    optim_node: fx.Node
    generate_output: bool
    param_outputs: List[fx.Node] = field(default_factory=list)
    grad_outputs: List[fx.Node] = field(default_factory=list)
    exp_avgs_outputs: List[fx.Node] = field(default_factory=list)
    exp_avg_sqs_outputs: List[fx.Node] = field(default_factory=list)
    max_exp_avg_sqs: List[fx.Node] = field(default_factory=list)

    def generate_outputs(self):

        def _generate_outputs(arg_idx, output_list):
            graph = self.optim_node.graph
            with graph.inserting_after(self.optim_node):
                optim_getitem = graph.call_function(operator.getitem, (self.optim_node, arg_idx))
            for i, arg in enumerate(self.optim_node.args[arg_idx]):
                with graph.inserting_after(optim_getitem):
                    updated_arg = graph.call_function(operator.getitem, (optim_getitem, i))
                with graph.inserting_after(updated_arg):
                    output_copy = graph.call_function(aten.copy_, (arg, updated_arg))
                output_list.append(output_copy)
        _generate_outputs(0, self.param_outputs)
        _generate_outputs(2, self.exp_avgs_outputs)
        _generate_outputs(3, self.exp_avg_sqs_outputs)

    def populate_outputs(self):

        def _populate_outputs(args_idx, output_list):
            optim_getitem = self.optim_node
            for user in self.optim_node.users:
                assert user.target == operator.getitem, f'The user of {self.optim_node} is not getitem.'
                if user.args[1] == args_idx:
                    optim_getitem = user
                    break
            assert optim_getitem != self.optim_node, f'Cannot find the getitem node for {self.optim_node}'
            output_list.extend([self.optim_node] * len(cast(List[fx.Node], self.optim_node.args[0])))
            for updated_arg in optim_getitem.users:
                assert updated_arg.target == operator.getitem, f'Unexpected node target {updated_arg.target}.'
                idx = updated_arg.args[1]
                output_copy = next(iter(updated_arg.users))
                assert str(output_copy.target).startswith('aten.copy_'), f'Unexpected node target {output_copy.target}.'
                output_list[idx] = output_copy
            for i, output in enumerate(output_list):
                assert output != self.optim_node, f'{i}th output is not replaced.'
            assert output_list, f'The output for {self.optim_node} is empty.'
        _populate_outputs(0, self.param_outputs)
        _populate_outputs(2, self.exp_avgs_outputs)
        _populate_outputs(3, self.exp_avg_sqs_outputs)

    def __post_init__(self):
        if self.param_outputs:
            return
        if self.generate_output:
            self.generate_outputs()
        else:
            self.populate_outputs()