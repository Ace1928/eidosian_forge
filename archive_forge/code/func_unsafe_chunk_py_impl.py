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
@aten.unsafe_chunk.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def unsafe_chunk_py_impl(tensor, chunks, dim=0) -> List[Tensor]:
    dim_size = tensor.size(dim)
    split_size = (dim_size + chunks - 1) // chunks
    if split_size == 0 and dim_size == 0:
        split_sizes = [split_size for _ in chunks]
        split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size)
        return torch.ops.aten.unsafe_split_with_sizes.default(tensor, split_sizes, dim)
    return torch.ops.aten.unsafe_split.Tensor(tensor, split_size, dim)