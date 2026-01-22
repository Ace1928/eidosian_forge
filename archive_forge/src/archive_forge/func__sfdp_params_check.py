import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_params_check(match):
    assert all((k in match.kwargs for k in ('query', 'key', 'value')))
    query = match.kwargs['query'].meta['val']
    key = match.kwargs['key'].meta['val']
    value = match.kwargs['value'].meta['val']
    if not query.dtype == key.dtype == value.dtype or not query.device == key.device == value.device:
        return False
    add_mask_node = filter_nodes(match.nodes, aten.add.Tensor)
    if len(add_mask_node) > 0:
        attn_mask_node = add_mask_node[0].args[1]
        if not hasattr(attn_mask_node, 'meta'):
            return False
        attn_mask = attn_mask_node.meta['val']
        if not isinstance(attn_mask, torch.Tensor) or not (attn_mask.dtype == query.dtype or attn_mask.dtype == torch.bool) or query.device != attn_mask.device:
            return False
    return True