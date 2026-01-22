import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
def _match_levels(v, from_levels, to_levels):
    view = []
    permute = []
    requires_view = False
    size = v.size()
    for t in to_levels:
        try:
            idx = from_levels.index(t)
            permute.append(idx)
            view.append(size[idx])
        except ValueError:
            view.append(1)
            requires_view = True
    if permute != list(range(len(permute))):
        v = v.permute(*permute)
    if requires_view:
        v = v.view(*view)
    return v