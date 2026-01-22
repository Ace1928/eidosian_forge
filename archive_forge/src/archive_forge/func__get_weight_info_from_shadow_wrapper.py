import torch
import torch.fx
from torch.fx import (
from torch.ao.ns.fx.utils import (
from torch.ao.ns.fx.ns_types import (
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.utils._pytree import tree_map
import collections
import copy
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
import operator
def _get_weight_info_from_shadow_wrapper(shadow_wrapper: torch.nn.Module):
    placeholders_seen = 0
    for shadow_n in shadow_wrapper.graph.nodes:
        if shadow_n.op != 'placeholder':
            continue
        placeholders_seen += 1
        if placeholders_seen != 2:
            continue
        assert len(shadow_n.users) == 1
        quant_node = next(iter(shadow_n.users.keys()))
        new_args: Any = None
        if quant_node.target == torch.quantize_per_channel:
            _weight, scale_node, zp_node, axis, dtype = quant_node.args
            scale_val = getattr_from_fqn(shadow_wrapper, scale_node.target)
            zp_val = getattr_from_fqn(shadow_wrapper, zp_node.target)
            new_args = (scale_val, zp_val, axis, dtype)
        else:
            assert quant_node.target == torch.quantize_per_tensor
            _weight, scale_node, zp_node, dtype = quant_node.args
            scale_val = getattr_from_fqn(shadow_wrapper, scale_node.target)
            zp_val = getattr_from_fqn(shadow_wrapper, zp_node.target)
            new_args = (scale_val, zp_val, dtype)
        return (quant_node.target, new_args)
    return None