from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
def _reduce_shard_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, reduce_op: c10d.ReduceOp.RedOpType, mesh_dim: int) -> torch.Tensor:
    """
        reduce and scatter a tensor on a mesh dimension
        """
    my_coordinate = mesh.get_coordinate()
    num_chunks = mesh.size(mesh_dim=mesh_dim)
    if my_coordinate is None:
        return tensor
    is_padded = tensor.size(self.dim) % num_chunks != 0
    if is_padded:
        scattered_list, pad_sizes = self._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)
        tensor = torch.cat(scattered_list, dim=self.dim)
    output = funcol.reduce_scatter_tensor(tensor, reduce_op.name, scatter_dim=self.dim, group=(mesh, mesh_dim))
    if is_padded:
        output = self._unpad_tensor(output, pad_sizes[my_coordinate[mesh_dim]])
    return output