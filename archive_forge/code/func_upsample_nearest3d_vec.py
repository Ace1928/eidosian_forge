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
@register_decomposition(aten.upsample_nearest3d.vec)
@aten.upsample_nearest3d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest3d.vec.py_impl(DispatchKey.Autograd)
def upsample_nearest3d_vec(input, output_size, scale_factors):
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scale_d = get_scale_value(scale_factors, 0)
    scale_h = get_scale_value(scale_factors, 1)
    scale_w = get_scale_value(scale_factors, 2)
    return aten.upsample_nearest3d.default(input, osize, scale_d, scale_h, scale_w)