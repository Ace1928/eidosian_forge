from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
def _unpad_tensor(self, tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    return tensor.narrow(self.dim, start=0, length=tensor.size(self.dim) - pad_size)