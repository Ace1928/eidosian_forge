import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
def _set_parameters_using_data_flow(self) -> None:
    """Deduce which Tensors are parameters.

        Consider the following code for the step of SGD with momentum
        (nesterov=False), where `d_p` is the gradient of `param` and `buf` is
        the momentum buffer.
        ```
          buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
          d_p = buf
          param.add_(d_p, alpha=-lr)
        ```
        Both `param` and `buf` take a gradient and perform an in-place update.

        The python tracer will inspect calls to `nn.Module.forward` and
        `optim.Optimizer.step` to extract parameter and optimizer state
        respectively (including parameters), so this is generally a non-issue.

        However as a fallback we can also exploit several properties of
        parameters to distinguish them from other model state.

        First, they are directly used in the forward pass. (At this point we
        haven't established which parts of the graph correspond to the forward
        pass but we can deduce enough to suffice.) Some mutable state such as
        batch norm moving averages also contribute to the forward pass, but
        optimizer state does not.

        Second, a parameter is by definition used to compute at least one
        gradient and depends on at least one gradient.
        """
    snapshot = self._category_snapshot()
    candidate_parameters: Set[TensorAndID] = set()
    candidate_fwd_tensors: Set[TensorAndID] = {i for i, category in snapshot.items() if category == Category.INPUT}
    for node in self._data_flow_graph.flow_nodes:
        inputs = {(key, value) for key, (_, value) in node.inputs.items()}
        if RecordScope.BACKWARD_FUNCTION not in get_scopes(node._event) and (not any((self._is_gradient(*i) for i in inputs))) and (not any((self._is_gradient(*i) for i in node.outputs.items()))) and candidate_fwd_tensors.intersection(inputs):
            candidate_fwd_tensors |= node.outputs.items()
            candidate_parameters |= inputs.difference(candidate_fwd_tensors)
    used_for_gradient: Set[TensorAndID] = set()
    for node in reversed(self._data_flow_graph.flow_nodes):
        if any((self._is_gradient(*i) or i in used_for_gradient for i in node.outputs.items())):
            for key, (_, version) in node.inputs.items():
                used_for_gradient.add((key, version))
    candidate_parameters.intersection_update(used_for_gradient)
    parameter_keys = {key.id for key, _ in candidate_parameters}
    parameter_keys &= self._any_version_depends_on_gradient()
    for key, _ in snapshot.keys():
        if key.id in parameter_keys:
            self._categories.set_by_id(key, Category.PARAMETER)