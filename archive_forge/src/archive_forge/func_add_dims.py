import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
def add_dims(t):
    if not isinstance(t, _Tensor):
        return
    for d in t.dims:
        dims_seen.record(d)