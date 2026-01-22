import math
import functools
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from typing import (
from typing_extensions import ParamSpec, Self, TypeAlias
import torch
import torch.utils.hooks as hooks
from torch.utils.hooks import RemovableHandle
from torch.utils._foreach_utils import (
from torch._utils import is_compiling
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
def _use_grad(self, *args, **kwargs):
    import torch._dynamo
    prev_grad = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(self.defaults['differentiable'])
        torch._dynamo.graph_break()
        ret = func(self, *args, **kwargs)
    finally:
        torch._dynamo.graph_break()
        torch.set_grad_enabled(prev_grad)
    return ret