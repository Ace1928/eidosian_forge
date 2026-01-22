import contextlib
import warnings
from typing import Dict, List, Optional
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._tensor.placement_types import DTensorSpec, Shard
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
def _set_pre_op_offset(self, spec: DTensorSpec) -> None:
    """Set the starting RNG offset for current device's local shard before actual
        op execution. The pre_op_offset value should start from the current RNG offset
        and increment by the size of local shard until it reaches the size of the whole
        DTensor. For different ranks that hold the same DTensor shard, their pre_op_offset
        will be the same.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we prepare the offset for running random ops.

        Returns:
            None

        .. warning::
            Note that, current implementation does not consider DTensor's continguity.

        Example:
            take a DTensor of shape [8, 16] as an example. Assume that the DTensor
            is placed on a device mesh with placements ([Shard(1), Replicate(), Shard(0)]),
            and the mesh is:
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
            ``spec.mesh.get_coordinate()`` provides the coordinate of the current rank
            in the mesh. For example, the coordinate of rank 5 is (1, 0, 1).

            Another concept to introduce besides rank coordinate is shard coordinate.
            Each rank holds a local shard of the DTensor. In the example, the DTensor
            is partitioned into 4 [4, 8] shards. The first shard has 2 replicas and
            rank 0 (coord (0, 0, 0)) and rank 2 (coord (0, 1, 0)) have 1 replica each.
            That being said, the local shard on rank 0 and rank 2 correspond to the same
            shard of the DTensor. To denote each DTensor shard, we use a shard coordinate
            (in the example, it will be a tuple (i, j) where shard (i, j) has the slice
            DTensor[4 * i : 4 * (i + 1), 8 * j : 8 * (j + 1)], 0 <= i < 2, 0 <= j < 2).

            Once we have rank coordinate and shard coordinate, we can calculate on each rank
            what shard of the DTensor the rank holds, with the help of dim_map. The dim_map
            of the above DTensor is [2, 0] so the shard coordinate of a rank with rank coord
            (x, y, z) is simply (z, x) by taking(rank_coord[dim_map[0]],rank_coord[dim_map[1]]).
            Following this calculation,
            rank 0 and rank 2 holds the shard of coord (0, 0);
            rank 1 and rank 3 holds the shard of coord (0, 1);
            rank 4 and rank 6 holds the shard of coord (1, 0);
            rank 5 and rank 7 holds the shard of coord (1, 1);

            The last value to calculate before obtaining the starting offset is the shard linear index.
            The starting offset for each rank will be its shard_linear_index * local_tensor_numel.
        """
    dtensor_shape = spec.shape
    mesh = spec.mesh
    dim_map = spec.dim_map
    coordinate = mesh.get_coordinate()
    assert coordinate is not None
    shard_coord = [coordinate[mesh_dim] if mesh_dim >= 0 else 0 for mesh_dim in dim_map]
    shard_size = [mesh.size(mesh_dim) if mesh_dim >= 0 else 1 for mesh_dim in dim_map]
    shard_linear_idx = self._calc_shard_linear_idx(shard_coord, shard_size)
    local_size_on_rank_0 = list(dtensor_shape)
    for idx, placement in enumerate(spec.placements):
        if isinstance(placement, Shard):
            mesh_dim_size = mesh.size(idx)
            shard_dim = placement.dim
            local_size_on_rank_0[shard_dim] = placement._local_shard_size_on_dim(dtensor_shape[shard_dim], mesh_dim_size, 0, return_offset=False)[0]
    from torch.distributed._tensor.ops.utils import prod
    local_size = prod(local_size_on_rank_0)
    current_offset = self.get_offset('parallel-rng')
    offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
    self.set_offset('parallel-rng', current_offset + offset_incr)