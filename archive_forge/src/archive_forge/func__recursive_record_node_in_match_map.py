import sys
import torch
from torch.fx.graph import (
from torch.ao.quantization.utils import Pattern
from .quantize_handler import (
from ..qconfig import (
from ..utils import (
from .graph_module import (
from torch.nn.utils.parametrize import type_before_parametrizations
from typing import Any, Dict, List, Callable, Optional, Tuple, Type, Set, Iterable
def _recursive_record_node_in_match_map(last_node, match_map, node_pattern, matched_node_pattern, pattern, match_value):
    if isinstance(node_pattern, Node):
        match_map[node_pattern.name] = (last_node, matched_node_pattern, pattern, match_value)
    elif not isinstance(node_pattern, Iterable):
        return
    else:
        for n in node_pattern:
            _recursive_record_node_in_match_map(last_node, match_map, n, matched_node_pattern, pattern, match_value)