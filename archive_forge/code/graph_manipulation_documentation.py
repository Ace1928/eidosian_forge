from typing import Any, Dict, List, NamedTuple, Optional
import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import (
from torch.fx.passes.shape_prop import ShapeProp
Given a node with node.dtype and node.shape, return its total size and its output size.
    total_size = weights + bias + output_size
    