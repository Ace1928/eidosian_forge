from .graph_module import GraphModule
from .graph import Graph
from .node import Node
from ._symbolic_trace import symbolic_trace
from ._compatibility import compatibility
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union
import torch
@compatibility(is_backward_compatible=False)
@dataclass
class ReplacedPatterns:
    anchor: Node
    nodes_map: Dict[Node, Node]
    replacements: List[Node]