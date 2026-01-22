import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph import Graph
from typing import Union, Dict, Any, Set
def _is_observed_module(module: Any) -> bool:
    return hasattr(module, 'meta') and '_observed_graph_module_attrs' in module.meta