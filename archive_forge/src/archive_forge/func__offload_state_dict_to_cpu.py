import math
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import AsyncCollectiveTensor
def _offload_state_dict_to_cpu(state_dict: Dict[str, Any], *, pg: Optional[dist.ProcessGroup]=None, device: Optional[torch.device]=None, ranks_only: Tuple[int, ...]=tuple()) -> Dict[str, Any]:
    return _iterate_state_dict(state_dict, lambda value, pg, device: value, lambda value, pg, device: value, pg=pg, device=device, cpu_offload=True, ranks_only=ranks_only)