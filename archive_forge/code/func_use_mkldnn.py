import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
def use_mkldnn(input, hx, params):
    if not torch._C._has_mkldnn:
        return False
    tensors = [input] + list(hx) + list(chain.from_iterable(params))
    devices = {t.device for t in tensors}
    if len(devices) != 1:
        return False
    device = devices.pop()
    if device != torch.device('cpu'):
        return False
    dtypes = {t.dtype for t in tensors}
    for dtype in dtypes:
        if dtype not in [torch.float, torch.bfloat16]:
            return False
    if input.requires_grad:
        return False
    has_projections = hx[0].size(2) != hx[1].size(2)
    if has_projections:
        return False
    return True