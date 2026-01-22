from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
def _to_replicate_tensor(self, local_tensor: torch.Tensor, size: torch.Size, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
    """
        This function all_gather all shards and return a tensor that
        is replicated on the previously sharded mesh dimension
        """
    my_coordinate = mesh.get_coordinate()
    num_chunks = mesh.size(mesh_dim=mesh_dim)
    if my_coordinate is None:
        return local_tensor
    full_chunk_size = (size[self.dim] + num_chunks - 1) // num_chunks
    chunk_sizes = [max(min(size[self.dim], full_chunk_size * (idx + 1)) - full_chunk_size * idx, 0) for idx in range(num_chunks)]
    pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]
    is_padded = size[self.dim] % num_chunks != 0
    pad_size = pad_sizes[my_coordinate[mesh_dim]]
    if pad_size > 0:
        local_tensor = self._pad_tensor(local_tensor, pad_size)
    local_tensor = local_tensor.contiguous()
    result = funcol.all_gather_tensor(local_tensor, gather_dim=self.dim, group=(mesh, mesh_dim))
    if is_padded:
        full_pad_size = sum(pad_sizes)
        result = self._unpad_tensor(result, full_pad_size)
    return result