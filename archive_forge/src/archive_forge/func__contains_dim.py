import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
def _contains_dim(input):
    from . import Dim
    for i in input:
        if isinstance(i, Dim):
            return True