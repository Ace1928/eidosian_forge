import functools
import warnings
from typing import Callable, Optional, Tuple, Union
import torch
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import _mesh_resources
def _create_1d_device_mesh(device_mesh: DeviceMesh, tp_mesh_dim: int=0) -> DeviceMesh:
    """
    Convert a N-D ``device_mesh`` into a 1D ``device_mesh`` for 1D Tensor Parallelism.

    Args:
        device_mesh (DeviceMesh):
            :class:``DeviceMesh`` object which describes the mesh topology
            of devices for the DTensor.
        tp_mesh_dim (int):
            the dimension of ``device_mesh`` where we perform
            Tensor Parallelism on.

    Return:
        device_mesh (DeviceMesh): 1-D :class:``DeviceMesh`` object that
            Tensor Parallelism operates on.
    """
    assert tp_mesh_dim < device_mesh.ndim and tp_mesh_dim >= -device_mesh.ndim, f'Expect tp_mesh_dim within range [{-device_mesh.ndim}, {device_mesh.ndim}), but found {tp_mesh_dim}.'
    if device_mesh.ndim == 1:
        return device_mesh
    cur_rank = device_mesh.get_rank()
    pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, tp_mesh_dim).reshape(-1, device_mesh.mesh.size(tp_mesh_dim))
    for mesh_1d in pg_ranks_by_dim:
        sub_mesh = DeviceMesh(device_mesh.device_type, mesh_1d, _init_process_groups=False)
        if cur_rank in mesh_1d:
            res_sub_mesh = sub_mesh
    res_sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[tp_mesh_dim]]
    return res_sub_mesh