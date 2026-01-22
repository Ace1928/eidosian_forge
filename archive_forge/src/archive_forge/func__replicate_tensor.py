from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
def _replicate_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
    """
        Replicate (broadcast) a torch.Tensor on a mesh dimension (use
        the first coordinate on the mesh dimension as source of truth)
        """
    my_coordinate = mesh.get_coordinate()
    if my_coordinate is None:
        return tensor.new_empty(0, requires_grad=tensor.requires_grad)
    tensor = tensor.contiguous()
    mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim)
    return tensor