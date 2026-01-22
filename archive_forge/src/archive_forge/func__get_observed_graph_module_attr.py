import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph import Graph
from typing import Union, Dict, Any, Set
def _get_observed_graph_module_attr(model: Union[torch.nn.Module, GraphModule], attr_name: str) -> Any:
    if hasattr(model, 'meta') and '_observed_graph_module_attrs' in model.meta:
        return getattr(model.meta['_observed_graph_module_attrs'], attr_name)
    return None