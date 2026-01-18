import abc
import typing as t
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from .shape_prop import TensorMetadata
from .tools_common import get_node_target, CALLABLE_NODE_OPS

        If a node has a name that is in the disallow set, reported it as non-supported.
        