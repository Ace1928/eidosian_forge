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
def _select_eps(self, graph_module: torch.fx.GraphModule, *args) -> Sequence[Tuple[str, Mapping[str, Any]]]:
    inferred_eps: Tuple[str, ...] = tuple()
    if self._options.infer_execution_providers:
        if (eps_from_args := _infer_ep_from_device(*args)):
            inferred_eps = eps_from_args
        elif (eps_from_graph_module := _infer_ep_from_graph_module(graph_module)):
            inferred_eps = eps_from_graph_module
    selected_eps = []
    for ep in (*(self._options.preferred_execution_providers or []), *_sort_eps(inferred_eps), *(self._options.default_execution_providers or _infer_default_eps())):
        if isinstance(ep, str):
            ep = (ep, {})
        elif isinstance(ep, tuple) and ep[1] is None:
            ep = (ep[0], {})
        if ep is not None and ep not in selected_eps:
            selected_eps.append(ep)
    return selected_eps