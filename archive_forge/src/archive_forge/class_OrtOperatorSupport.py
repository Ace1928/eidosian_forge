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
class OrtOperatorSupport(OperatorSupport):
    """Operator support for ONNXRuntime backend.

    It has two-level of support decision. One is via support_dict and the other one
    is via extra_support_dict. The logic of using support_dict is implemented in
    OrtOperatorSupport and extra_support_dict is used by OperatorSupport.is_node_supported.
    """

    def __init__(self, support_dict: Set[Any], extra_support_dict: Dict[str, Any]):
        super().__init__(extra_support_dict)
        self._onnx_support_dict = support_dict

    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        if node.op not in CALLABLE_NODE_OPS:
            return False
        if node.op == 'call_function' and node.target in self._onnx_support_dict:
            logger.warning('support_dict supports node.target: %s (type: %s)', node.target, type(node.target))
            return True
        logger.warning("support_dict doesn't support node.target: %s (type: %s)", node.target, type(node.target))
        if super().is_node_supported(submodules, node):
            logger.warning('extra_support_dict supports node.target: %s (type: %s)', node.target, type(node.target))
            return True
        logger.warning("extra_support_dict doesn't supports node.target: %s (type: %s)", node.target, type(node.target))
        return False