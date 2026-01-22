import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import LoggingTensor
class SubclassInfo:
    __slots__ = ['name', 'create_fn', 'closed_under_ops']

    def __init__(self, name, create_fn, closed_under_ops=True):
        self.name = name
        self.create_fn = create_fn
        self.closed_under_ops = closed_under_ops