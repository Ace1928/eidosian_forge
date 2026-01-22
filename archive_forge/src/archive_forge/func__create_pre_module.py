import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
def _create_pre_module(self, name):

    class PopState(torch.autograd.Function):

        @staticmethod
        def forward(ctx, *args):
            self.parents.append(name)
            args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
            return args

        @staticmethod
        def backward(ctx, *grad_outs):
            assert self.parents[-1] == name
            self.parents.pop()
            return grad_outs
    return PopState.apply