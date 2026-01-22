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
def _is_matchable(self, node: Node) -> bool:
    if node.op == 'call_function':
        return node.target not in self.non_matchable_functions
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        target_mod = getattr_from_fqn(self.gm, node.target)
        return not any((isinstance(target_mod, t) for t in self.non_matchable_modules))
    elif node.op == 'call_method':
        return node.target not in self.non_matchable_methods
    else:
        return False