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
def _compute_upsample_nearest_indices(input, output_size, scales, exact=False):
    indices = []
    num_spatial_dims = len(output_size)
    offset = 0.5 if exact else 0.0
    for d in range(num_spatial_dims):
        osize = output_size[d]
        isize = input.shape[-num_spatial_dims + d]
        scale = isize / (isize * scales[d]) if scales[d] is not None else isize / osize
        output_indices = torch.arange(osize, dtype=torch.float32, device=input.device)
        input_indices = ((output_indices + offset) * scale).to(torch.int64)
        for _ in range(num_spatial_dims - 1 - d):
            input_indices = input_indices.unsqueeze(-1)
        indices.append(input_indices)
    return tuple(indices)