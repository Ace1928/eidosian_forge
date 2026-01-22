import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
def _infer_ep_from_graph_module(graph_module: torch.fx.GraphModule) -> Tuple[str, ...]:
    """Return the all valid devices (i.e., GPU or CPU) among outputs of this torch.fx.GraphModule."""
    flattened_output_args, _ = _pytree.tree_flatten(_extract_graph_module_outputs(graph_module))
    selected_output_args = [output_arg.meta['val'] for output_arg in flattened_output_args if hasattr(output_arg, 'meta') and 'val' in output_arg.meta]
    return _infer_ep_from_device(*selected_output_args)