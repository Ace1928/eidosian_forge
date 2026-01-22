import contextlib
import warnings
from typing import Dict, List, Optional
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._tensor.placement_types import DTensorSpec, Shard
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
class OffsetBasedRNGTracker(RNGStateTracker):
    """
    This subclass of `RNGStateTracker` defines the default policy of how RNG states
    should be shared and synchronized among all ranks to respect the semantics of DTensor
    random operators.
    """

    def __init__(self, device_type: str='cuda'):
        super().__init__(device_type)
        rng_state = self._device_handle.get_rng_state().to(device_type)
        dist.broadcast(rng_state, 0)
        self.rng_states['parallel-rng'] = rng_state.to('cpu')

    def _manual_seed(self, parallel_seed: int) -> None:
        self.set_seed('parallel-rng', parallel_seed)

    @contextlib.contextmanager
    def _distribute_region(self, spec: DTensorSpec):
        if not self.rng_state_is_sync('parallel-rng'):
            raise RuntimeError('OffsetBasedRNGTracker requires the random state to be synchronized before entering into a distribute region!')
        if self.distribute_region_enabled:
            old_offset = self.get_offset('parallel-rng')
            self._set_pre_op_offset(spec)
            with torch.random.fork_rng(self._devices, device_type=self._device_type):
                self._device_handle.set_rng_state(self.rng_states['parallel-rng'])
                try:
                    yield
                finally:
                    self._set_post_op_offset(spec, old_offset)
        else:
            yield

    def get_offset(self, name: str) -> int:
        if name not in self.rng_states:
            raise RuntimeError(f'{self.__class__.__name__} does not have random state for {name}')
        offset_tensor = self.rng_states[name][8:].view(dtype=torch.int64)
        return int(offset_tensor.item())

    def set_offset(self, name: str, offset: int) -> None:
        if name not in self.rng_states:
            raise RuntimeError(f'{self.__class__.__name__} does not have random state for {name}')
        seed_tensor = self.rng_states[name][0:8]
        offset_tensor = torch.tensor([offset]).view(torch.uint8)
        self.rng_states[name] = torch.cat([seed_tensor, offset_tensor])

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

    def _set_post_op_offset(self, spec: DTensorSpec, old_offset: int) -> None:
        """Sets the RNG to a synchronized state after running the local random op. Every
        rank should set its RNG offset to `old_offset + DTensor.numel()` where old_offset is
        the offset before calling `set_pre_op_offset` i.e. the offset before running DTensor
        random ops.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we post-process the offset for running random ops.

        Returns:
            None
        """
        dtensor_shape = spec.shape
        from torch.distributed._tensor.ops.utils import prod
        numel = prod(dtensor_shape)
        numel = (numel + 3) // 4 * 4
        self.set_offset('parallel-rng', old_offset + numel)

    def _calc_shard_linear_idx(self, shard_coord: List[int], shard_size: List[int]) -> int:
        shard_linear_idx = 0
        shard_coord_stride = 1
        for idx, size in zip(reversed(shard_coord), reversed(shard_size)):
            shard_linear_idx += idx * shard_coord_stride
            shard_coord_stride *= size
        return shard_linear_idx