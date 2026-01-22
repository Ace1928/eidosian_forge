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
class GraphMatchingException(Exception):
    """
    Exception raised when two graphs cannot be matched.
    """
    pass