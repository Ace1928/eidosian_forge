import collections
import enum
import torch
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from torch.ao.quantization.utils import getattr_from_fqn
from .ns_types import NSSubgraph, NSNodeTargetType
from .mappings import (
from .pattern_utils import (
from torch.ao.quantization import (
from typing import Dict, Tuple, List, Optional, Set, Any
def _get_node_target_type(node: Node, gm: GraphModule) -> Optional[NSNodeTargetType]:
    if node.op in ('call_function', 'call_method'):
        return node.target
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        mod = getattr_from_fqn(gm, node.target)
        return type(mod)
    return None