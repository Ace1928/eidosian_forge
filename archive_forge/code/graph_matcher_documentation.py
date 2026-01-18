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

        Adds all of the nodes in this arg to the stack, properly navigating
        through list, dicts and tuples.
        