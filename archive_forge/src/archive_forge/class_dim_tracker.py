import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
class dim_tracker:

    def __init__(self):
        self.dims = llist()
        self.count = []

    def record(self, d):
        if d not in self.dims:
            self.dims.append(d)
            self.count.append(1)

    def __getitem__(self, d):
        return self.count[self.dims.index(d)]