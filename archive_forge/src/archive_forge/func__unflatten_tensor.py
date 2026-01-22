from functools import partial
from typing import no_type_check, Optional, Tuple
import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import DTensorSpec
@no_type_check
def _unflatten_tensor(tensor, spec, *, device_handle=None, compute_stream=None):
    result = DTensor.from_local(tensor, spec.mesh, spec.placements, run_check=False)
    if tensor.requires_grad:
        tensor.register_hook(partial(sync_grad_hook, device_handle=device_handle, compute_stream=compute_stream))
    return result