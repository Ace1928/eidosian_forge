import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
def _pad3d_common(input, padding, *, is_reflection):
    dim_w = 3
    dim_h = 2
    dim_d = 1
    dim_plane = 0
    _padding_check_valid_input(input, padding, dim=3)
    batch_mode = input.ndim == 5
    if batch_mode:
        nbatch = input.size(0)
        dim_w += 1
        dim_h += 1
        dim_d += 1
        dim_plane += 1
    pad_l, pad_r, pad_t, pad_b, pad_f, pad_bk = padding
    nplane = input.size(dim_plane)
    input_d = input.size(dim_d)
    input_h = input.size(dim_h)
    input_w = input.size(dim_w)
    output_d = input_d + pad_f + pad_bk
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r
    if is_reflection:
        torch._check(pad_l < input_w and pad_r < input_w, lambda: f'Argument #4: Padding size should be less than the corresponding input dimension, but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}')
        torch._check(pad_t < input_h and pad_b < input_h, lambda: f'Argument #6: Padding size should be less than the corresponding input dimension, but got: padding ({pad_t}, {pad_b}) at dimension {dim_h} of input {input.shape}')
        torch._check(pad_f < input_d and pad_bk < input_d, lambda: f'Argument #8: Padding size should be less than the corresponding input dimension, but got: padding ({pad_f}, {pad_bk}) at dimension {dim_d} of input {input.shape}')
    torch._check(output_w >= 1 or output_h >= 1 or output_d >= 1, lambda: f'input (D: {input_d} H: {input_h} W: {input_w}) is too small. Calculated output D: {output_d} H: {output_h} W: {output_w}')
    if batch_mode:
        return input.new_empty((nbatch, nplane, output_d, output_h, output_w))
    else:
        return input.new_empty((nplane, output_d, output_h, output_w))