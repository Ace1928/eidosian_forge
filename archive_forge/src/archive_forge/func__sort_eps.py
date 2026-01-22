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
def _sort_eps(eps: Tuple[str, ...]) -> Tuple[str, ...]:
    """Sort execution providers in eps based on pre-set priority."""

    def get_execution_provider_priority(ep: str) -> int:
        if ep == 'CPUExecutionProvider':
            return 2
        if ep == 'CUDAExecutionProvider':
            return 1
        return 0
    unique_eps = set(eps)
    return tuple(sorted(unique_eps, key=get_execution_provider_priority, reverse=True))