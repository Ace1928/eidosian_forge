from dataclasses import dataclass
from typing import Callable, cast, Dict, Iterable, Optional, Sequence, Set, Tuple, Union
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.api import Shard
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import DTensorSpec, Placement, Replicate
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
def view_groups(from_size: Shape, to_size: Shape) -> DimMap:
    """
    Decompose a reshape operation into forwarding, flattening, or splitting dimensions for each output dimension.

    A view or reshape operation can be decomposed into a set of 3 types of smaller operations:
    1) Forward a dimension from input to output
    2) Flatten a set of dimensions into a single dimension
    3) Split one dimension into multiple dimensions

    view_groups identifies these operations and returns, for each output dimension, what
    is operation was performed in the input dimension. For example:

        view_groups([2, 3, 4], [2, 12]) -> (
            InputDim(0),
            Flatten((InputDim(1), InputDim(2)))
        )

    - ouptut dimension 0 maps to input dimension 0
    - output dimension 1 maps to a flattened input dimensions 1 and 2


        view_groups([2, 3], [3, 2]) -> (
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
        )

    - in the above, input is flattened into a single dimension and then split
      into two separate dimensions with different sizes from the input.
    """
    from_nelem = prod(from_size)
    to_size = infer_size(from_nelem, normalize_sizes(to_size))
    assert from_nelem == prod(to_size), 'Total view shape does not add up'
    from_idx = 0
    to_idx = 0
    from_len = len(from_size)
    to_len = len(to_size)
    result_pp = []
    while from_idx < from_len or to_idx < to_len:
        from_group_dim, to_group_shape = ([], [])
        if from_idx >= from_len:
            f = 1
        else:
            f = from_size[from_idx]
            from_group_dim.append(from_idx)
            from_idx += 1
        if to_idx >= to_len:
            t = 1
        else:
            t = to_size[to_idx]
            to_group_shape.append(t)
            to_idx += 1
        if f == 1 and t != 1:
            to_idx -= 1
            to_group_shape = []
        elif f != 1 and t == 1:
            from_idx -= 1
            from_group_dim = []
        else:
            while f != t:
                if f < t:
                    nf = from_size[from_idx]
                    from_group_dim.append(from_idx)
                    from_idx += 1
                    f *= nf
                else:
                    nt = to_size[to_idx]
                    to_group_shape.append(nt)
                    to_idx += 1
                    t *= nt
        if len(to_group_shape) > 0:
            flattened = Flatten.new(tuple((InputDim(fi) for fi in from_group_dim if from_size[fi] > 1)))
            result_pp += [Split.new(flattened, tuple(to_group_shape), i) for i in range(len(to_group_shape))]
    return tuple(result_pp)