from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
def _module_stack_meta_from_node(node: torch.fx.Node) -> _ModuleStackMeta:
    return _ModuleStackMeta(node.meta.get('nn_module_stack'))