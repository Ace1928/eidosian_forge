import functools
import operator
from functools import reduce
from typing import Any, Tuple
import torch
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from .. import ir
from ..lowering import lowerings as L
from ..pattern_matcher import (
from ..virtualized import ops
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
def _is_valid_binary(match, fn):
    binary_nodes = filter_nodes(match.nodes, fn)
    if len(binary_nodes) < 1:
        return False
    if any((not (hasattr(n.args[0], 'meta') and isinstance(n.args[0].meta.get('val', None), torch.Tensor)) or not (hasattr(n.args[1], 'meta') and isinstance(n.args[1].meta.get('val', None), torch.Tensor)) for n in binary_nodes)):
        return False
    if any((get_arg_value(n, 2, kwarg_name='alpha') != 1.0 and get_arg_value(n, 2, kwarg_name='alpha') is not None for n in binary_nodes)):
        return False
    if any((n.args[0].meta['val'].size() != n.args[1].meta['val'].size() or n.args[0].meta['val'].device != n.args[1].meta['val'].device or n.args[0].meta['val'].dtype != n.args[1].meta['val'].dtype for n in binary_nodes)):
        return False
    if any((n.args[0] == n.args[1] for n in binary_nodes)):
        return False
    return True