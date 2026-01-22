from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
def _replace_tuple_with_list(spec: pytree.TreeSpec) -> pytree.TreeSpec:
    _type = list if spec.type == tuple else spec.type
    return pytree.TreeSpec(_type, spec.context, list(map(_replace_tuple_with_list, spec.children_specs)))