import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def _get_leaf_node(self, module: torch.nn.Module, node: torch.fx.Node) -> torch.nn.Module:
    py_obj = module
    assert isinstance(node.target, str)
    atoms = node.target.split('.')
    for atom in atoms:
        if not hasattr(py_obj, atom):
            raise RuntimeError(str(py_obj) + ' does not have attribute ' + atom + '!')
        py_obj = getattr(py_obj, atom)
    return py_obj