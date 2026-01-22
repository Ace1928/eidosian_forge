from __future__ import annotations  # type: ignore[attr-defined]
from dataclasses import dataclass
from typing import (
import copy
import warnings
from functools import reduce
import weakref
import threading
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.distributed import distributed_c10d
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed._shard.sharding_spec.api import (
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard._utils import (
from .metadata import TensorProperties, ShardedTensorMetadata
from .shard import Shard
from .reshard import reshuffle_local_shard, reshard_local_shard
from .utils import (
from torch.distributed.remote_device import _remote_device
from torch.utils import _pytree as pytree
@classmethod
def _init_from_local_shards_and_global_metadata(cls, local_shards: List[Shard], sharded_tensor_metadata: ShardedTensorMetadata, process_group=None, init_rrefs=False, sharding_spec=None) -> ShardedTensor:
    """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
    process_group = process_group if process_group is not None else distributed_c10d._get_default_group()
    current_rank = dist.get_rank(process_group)
    shards_metadata = sharded_tensor_metadata.shards_metadata
    local_shard_metadatas = []
    for shard_metadata in shards_metadata:
        rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)
        if current_rank == rank:
            local_shard_metadatas.append(shard_metadata)
    if len(local_shards) != len(local_shard_metadatas):
        raise RuntimeError(f'Number of local shards ({len(local_shards)}) does not match number of local shards metadata in sharded_tensor_metadata ({len(local_shard_metadatas)}) on rank ({current_rank}) ')
    shards_metadata = sharded_tensor_metadata.shards_metadata
    tensor_properties = sharded_tensor_metadata.tensor_properties
    if len(shards_metadata) == 0:
        raise ValueError('shards_metadata must not be empty!')
    if tensor_properties.layout != torch.strided:
        raise ValueError('Only torch.strided layout is currently supported')
    if sharding_spec is None:
        spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
    else:
        spec = sharding_spec
    sharded_tensor = ShardedTensor.__new__(ShardedTensor, spec, sharded_tensor_metadata.size, dtype=tensor_properties.dtype, layout=tensor_properties.layout, pin_memory=tensor_properties.pin_memory, requires_grad=tensor_properties.requires_grad)

    def _raise_if_mismatch(expected, actual, prop_name, rank, is_property=False):
        tensor_property_or_metadata = 'tensor property' if is_property else 'local ShardMetadata'
        if expected != actual:
            raise ValueError(f"Local shards' tensor {prop_name} property is incompatible with {tensor_property_or_metadata} on rank {rank}: {tensor_property_or_metadata} {prop_name}={expected}, local shard tensor {prop_name}={actual}.")
    for shard in local_shards:
        shard_meta = shard.metadata
        local_shard_tensor = shard.tensor
        placement = shard_meta.placement
        assert placement is not None, 'Must specify placement for `Shard`!'
        rank = placement.rank()
        local_device = placement.device()
        _raise_if_mismatch(tensor_properties.layout, local_shard_tensor.layout, 'layout', rank, True)
        if not local_shard_tensor.is_contiguous():
            raise ValueError('Only torch.contiguous_format memory_format is currently supported')
        _raise_if_mismatch(shard_meta.shard_sizes, list(local_shard_tensor.size()), 'size', rank)
        _raise_if_mismatch(tensor_properties.pin_memory, local_shard_tensor.is_pinned(), 'pin_memory', rank, True)
        _raise_if_mismatch(local_device, local_shard_tensor.device, 'device', rank)
        _raise_if_mismatch(tensor_properties.dtype, local_shard_tensor.dtype, 'dtype', rank, True)
        _raise_if_mismatch(tensor_properties.requires_grad, local_shard_tensor.requires_grad, 'requires_grad', rank, True)
    validate_non_overlapping_shards_metadata(shards_metadata)
    check_tensor(shards_metadata, list(sharded_tensor_metadata.size))
    sharded_tensor._local_shards = local_shards
    sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)
    sharded_tensor._post_init()
    return sharded_tensor