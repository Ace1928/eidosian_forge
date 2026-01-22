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
def _extract_parameters_and_gradients(node: _ProfilerEvent) -> Iterator[Tuple[Optional[TensorKey], Optional[TensorKey]]]:
    children = node.children
    if node.typed[0] == _EventType.TorchOp and node.typed[1].scope == RecordScope.BACKWARD_FUNCTION and (node.name == 'torch::autograd::AccumulateGrad') and children and (children[0].typed[0] == _EventType.TorchOp) and (children[0].name in ('aten::detach', 'aten::add_')) and children[0].typed[1].inputs and isinstance(children[0].typed[1].inputs[0], _TensorMetadata):
        yield (None, TensorKey.from_tensor(children[0].typed[1].inputs[0]))
    elif node.typed[0] == _EventType.PyCall:
        typed_fields = node.typed[1]
        assert typed_fields.module is None or typed_fields.optimizer is None
        if typed_fields.module is not None:
            for _, p, p_grad in typed_fields.module.parameters:
                yield (TensorKey.from_tensor(p), TensorKey.from_tensor(p_grad))
        if typed_fields.optimizer is not None:
            for p, p_grad, _ in typed_fields.optimizer.parameters:
                yield (TensorKey.from_tensor(p), TensorKey.from_tensor(p_grad))