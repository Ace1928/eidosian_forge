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
@compatibility(is_backward_compatible=False)
def torch_compile_backend(graph_module: torch.fx.GraphModule, args, *, options: Optional[Union[OrtBackendOptions, Mapping[str, Any]]]=None):
    return OrtBackend.get_cached_instance_for_options(options)(graph_module, args)