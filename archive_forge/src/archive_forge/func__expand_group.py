import warnings
import sys
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import Tuple, Union, List, Optional, cast, TYPE_CHECKING
from . import _functional_collectives_impl as fun_col_impl
from ._functional_collectives_impl import _register_tensor_wrapper
from torch.fx.experimental.proxy_tensor import (
from torch._custom_ops import impl_abstract
from torch.distributed.distributed_c10d import (
def _expand_group(group: RANK_TYPES, tag: str='') -> Tuple[str, List[int], int]:
    """
    _expand_group desugars the different RANK_TYPES types into a canonical format that is traceable.

    By having this be part of the explicit eager codepath, we avoid having to specialize behavior inside
    torchdynamo and can still interoperate with processgroup objects or other untraceable forms.
    """
    import torch.distributed._tensor as dt
    if TYPE_CHECKING:

        def cast_listlistint(x):
            return cast(List[List[int]], x)

        def cast_listint(x):
            return cast(List[int], x)
    else:

        def cast_listlistint(x):
            return x

        def cast_listint(x):
            return x
    rankset: List[int]
    if isinstance(group, list):
        if isinstance(group[0], list):
            nested_list = cast_listlistint(group)
            rankset = []
            group_size = -1
            for rs in nested_list:
                rankset.extend(rs)
                if group_size != -1 and group_size != len(rs):
                    raise ValueError(f'group sizes must be identical found {group_size} and {len(rs)}')
                group_size = len(rs)
        else:
            rankset = cast_listint(group)
            group_size = len(rankset)
    elif isinstance(group, dist.ProcessGroup):
        rankset = dist.get_process_group_ranks(group)
        group_size = len(rankset)
        tag = tag or c10d._get_group_tag(group)
    elif isinstance(group, dt.DeviceMesh):
        assert group.ndim == 1, 'Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D'
        tag, rankset = group._dim_group_infos[0]
        group_size = len(rankset)
    elif isinstance(group, tuple):
        if len(group) == 2 and isinstance(group[0], dt.DeviceMesh) and isinstance(group[1], int):
            dmesh = group[0]
            dim = group[1]
            tag, rankset = dmesh._dim_group_infos[dim]
            group_size = len(rankset)
        else:
            raise ValueError('Invalid tuple for group must be (DeviceMesh, int)')
    else:
        raise ValueError('Invalid type for group, must be one of List, Processgroup, DeviceMesh or (DeviceMesh, int).')
    return (tag, rankset, group_size)