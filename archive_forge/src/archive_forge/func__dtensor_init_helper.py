from typing import Optional, Sequence
import torch
import torch.distributed._tensor.ops
import torch.distributed._tensor.random as random
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.api import distribute_module, distribute_tensor, DTensor
from torch.distributed._tensor.ops.utils import normalize_to_torch_size
from torch.distributed._tensor.placement_types import Placement, Replicate, Shard
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh, init_device_mesh
def _dtensor_init_helper(init_op, size: torch.Size, device_mesh=None, placements=None, **kwargs) -> DTensor:
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    kwargs['device'] = device_mesh.device_type
    placements = placements or tuple((Replicate() for _ in range(device_mesh.ndim)))
    assert device_mesh.ndim == len(placements), 'mesh dimension does not match the length of placements'
    assert kwargs['layout'] == torch.strided, 'layout value not supported!'
    torch_stride = torch._prims_common.make_contiguous_strides_for(size)
    local_shape = compute_local_shape(size, device_mesh, placements)
    if init_op == torch.full:
        fill_value = kwargs.pop('fill_value', 0)
        local_tensor = init_op(local_shape, fill_value, **kwargs)
    elif init_op == torch.rand or init_op == torch.randn:
        dtype = kwargs.get('dtype', torch.get_default_dtype())
        from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
        tensor_meta = TensorMeta(size, (0,), dtype)
        spec = DTensorSpec(device_mesh, placements, tensor_meta=tensor_meta)
        if random.is_rng_supported_mesh(device_mesh) and (not random._rng_tracker):
            random._rng_tracker = random.OffsetBasedRNGTracker()
        assert random._rng_tracker is not None
        with random._rng_tracker._distribute_region(spec):
            local_tensor = init_op(local_shape, **kwargs)
    else:
        local_tensor = init_op(local_shape, **kwargs)
    return DTensor(local_tensor=local_tensor, device_mesh=device_mesh, placements=tuple(placements), shape=size, dtype=local_tensor.dtype, stride=torch_stride, requires_grad=kwargs['requires_grad'])