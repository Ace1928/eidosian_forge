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
def upsample_common_check(input_size, output_size, num_spatial_dims):
    torch._check(len(output_size) == num_spatial_dims, lambda: f'It is expected output_size equals to {num_spatial_dims}, but got size {len(output_size)}')
    expected_input_dims = num_spatial_dims + 2
    torch._check(len(input_size) == expected_input_dims, lambda: f'It is expected input_size equals to {expected_input_dims}, but got size {len(input_size)}')
    torch._check(all((s > 0 for s in input_size[2:])) and all((s > 0 for s in output_size)), lambda: f'Input and output sizes should be greater than 0, but got input size {input_size} and output size {output_size}')
    nbatch, channels = input_size[:2]
    return (nbatch, channels, *output_size)