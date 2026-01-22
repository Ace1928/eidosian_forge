import math
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import AsyncCollectiveTensor
def _gather_state_dict(state_dict: Dict[str, Any], *, pg: Optional[dist.ProcessGroup]=None, device: Optional[torch.device]=None, cpu_offload: bool=False, ranks_only: Tuple[int, ...]=tuple()) -> Dict[str, Any]:
    """
    Given a state_dict, this API gathers all the ShardedTensors or DTensors in
    the state_dict.


    Args:
        state_dict (Dict[str, Any]): the target sharded state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        device: (Optional[torch.device]): the device that is used to
            perform allgather for ShardedTensor. Note that gathering a DTensor
            will use the DeviceMesh. So this argument will be ignored when
            gathering a DTensor.
        cpu_offload (bool): whether to offload the tensors to CPU memory. The
            default value is False.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.

    Returns:
        The gathered state dictionary.
    """

    def sharded_tensor_func(value, pg, device):
        cpu_device = torch.device('cpu')
        output_tensor = _all_gather_sharded_tensor(value, pg, device)
        local_shard_device = value.local_shards()[0].tensor.device if value.local_shards() else cpu_device
        if output_tensor.device != local_shard_device:
            value = output_tensor.to(local_shard_device)
        else:
            value = output_tensor
        return value

    def dtensor_func(value, pg, device):
        if value.device != value.device_mesh.device_type:
            value = value.to(value.device_mesh.device_type)
        placements = [Replicate() for _ in value.placements]
        value = value.redistribute(device_mesh=value.device_mesh, placements=placements)
        value = value.to_local()
        if isinstance(value, AsyncCollectiveTensor):
            value = value.wait()
        return value
    return _iterate_state_dict(state_dict, sharded_tensor_func, dtensor_func, pg=pg, device=device, cpu_offload=cpu_offload, ranks_only=ranks_only)