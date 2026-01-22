import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def _typename(self, target: Any) -> str:
    if isinstance(target, torch.nn.Module):
        ret = torch.typename(target)
    elif isinstance(target, str):
        ret = target
    else:
        ret = _get_qualified_name(target)
    return ret.replace('{', '\\{').replace('}', '\\}')