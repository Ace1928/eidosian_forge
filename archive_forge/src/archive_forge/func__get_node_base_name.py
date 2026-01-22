from __future__ import annotations
import collections
import re
from typing import Callable, Dict, Optional, Tuple
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.onnx._internal import _beartype
def _get_node_base_name(node_name: str) -> Tuple[str, Optional[int]]:
    pattern = '(.*)\\.(\\d+)'
    match = re.match(pattern, node_name)
    if match is not None:
        base_name, count_str = match.groups()
        return (base_name, int(count_str))
    return (node_name, None)