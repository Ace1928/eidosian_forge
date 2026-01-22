import abc
import typing as t
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from .shape_prop import TensorMetadata
from .tools_common import get_node_target, CALLABLE_NODE_OPS
def _any_chain(submods, node) -> bool:
    return any((x.is_node_supported(submods, node) for x in op_support))