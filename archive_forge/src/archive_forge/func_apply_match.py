from torch.fx import (
from torch.fx.graph import Graph
from .match_utils import (
from .pattern_utils import (
from ..backend_config import (
from ..backend_config.utils import (
from .custom_config import FuseCustomConfig
from .fuse_handler import (
from typing import Any, Callable, Dict, List, Tuple, Union
import warnings
from torch.ao.quantization.utils import Pattern, NodePattern
def apply_match(pattern, node, match, matched_node_pattern, node_to_subpattern):
    if isinstance(pattern, tuple):
        s, *args = pattern
        current_node_pattern: List[Node] = []
        apply_match(s, node, match, current_node_pattern, node_to_subpattern)
        for subpattern, arg in zip(args, node.args):
            apply_match(subpattern, arg, match, current_node_pattern, node_to_subpattern)
        matched_node_pattern.append(tuple(current_node_pattern))
    elif node.name not in match_map:
        matched_node_pattern.append(node)
        if pattern is not MatchAllNode:
            node_to_subpattern[node] = pattern
            root_node, pattern, handler = match
            match_map[node.name] = (root_node, pattern, matched_node_pattern, handler, node_to_subpattern)