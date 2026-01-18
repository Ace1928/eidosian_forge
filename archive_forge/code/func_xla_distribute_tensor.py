import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Placement, Replicate
from torch.distributed.device_mesh import DeviceMesh
@with_xla
def xla_distribute_tensor(tensor: torch.Tensor, device_mesh: DeviceMesh, placements: Optional[Sequence[Placement]]=None) -> 'XLAShardedTensor':
    """
    Distribute a torch.Tensor to the `device_mesh` according to the `placements`
    specified. The rank of `device_mesh` and `placements` must be the same.

    Args:
        tensor (torch.Tensor): torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use `torch.chunk`
            semantic to shard the tensor and scatter the shards.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as `device_mesh.ndim`. If not specified, we will
            by default replicate the tensor across the `device_mesh` from the
            first rank of each dimension of the `device_mesh`.

    Returns:
        A :class:`XLAShardedTensor` object

    .. note:: We return a XLAShardedTensor with a global view and access to local shards.
    The successive ops would be programmed as if on a single-device and without calling
    any explicit collective ops. The actual sharded computation on the sharding annotated tensor
    happens lazily, is transparent to the user. In the future, we will introduce
    a new DTensor type for this kind of programming-mode (single-controller) and return.
    """
    dt_mesh = device_mesh
    assert dt_mesh.device_type == 'xla'
    xla_mesh = convert_to_xla_mesh(dt_mesh)
    assert xla_mesh.mesh_shape == tuple(dt_mesh.mesh.size())
    if not tensor.is_meta:
        tensor = tensor.to(dt_mesh.device_type)
    if placements is None:
        placements = [Replicate() for _ in range(dt_mesh.ndim)]
    assert len(placements) == dt_mesh.ndim, '`placements` must have the same length as `device_mesh.ndim`! '
    f'Found placements length: {len(placements)}, and device_mesh.ndim: {dt_mesh.ndim}.'
    partition_spec = convert_to_xla_partition_spec(tensor, placements)
    assert len(tensor.shape) == len(partition_spec), '`partition_spec` from `placements` must have the same length as `tensor.length`! '
    f'Found tensor shape length: {len(tensor.shape)}, and partition_spec length: {len(partition_spec)}.'
    global_tensor = tensor
    if type(tensor).__name__ == 'DTensor':
        raise ValueError('Cannot distribute a DTensor with local tensor on xla devices.The input tensor must be global.')
    if type(tensor).__name__ == 'XLAShardedTensor':
        sharding_type = tensor.sharding_type
        assert sharding_type is None or sharding_type == ShardingType.REPLICATED, 'XLAShardedTensor `tensor` is already annotated with non-replication sharding. '
        'Clear the existing sharding annotation first, by callling torch_xla.experimental.xla_sharding.clear_sharding API.'
        global_tensor = tensor.global_tensor
    assert global_tensor is not None, 'distributing a tensor should not be None'
    xla_tensor = mark_sharding(global_tensor, xla_mesh, partition_spec)
    return xla_tensor