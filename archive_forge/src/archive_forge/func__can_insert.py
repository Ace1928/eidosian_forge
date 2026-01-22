import torch
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from .utils import (
from .ns_types import (
from torch.ao.ns.fx.mappings import (
from torch.ao.quantization.observer import _is_activation_post_process
from typing import Dict, Tuple, Callable, List, Any, Union, Optional, Set
def _can_insert(node_a_arg, gm_a):
    if isinstance(node_a_arg, Node):
        arg_a = return_first_non_observer_node(node_a_arg, gm_a)
        if arg_a.op == 'call_method':
            return arg_a.target in ('dequantize', 'to')
        elif arg_a.op == 'get_attr':
            return True
        else:
            return False
    elif isinstance(node_a_arg, (list, tuple)):
        for el in node_a_arg:
            if not isinstance(el, Node):
                return False
    return True