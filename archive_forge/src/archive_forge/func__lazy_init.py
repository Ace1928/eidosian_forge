import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
def _lazy_init(device: torch.device, dtype: torch.dtype, group: dist.ProcessGroup, num_stripes: int) -> Optional[_FusedSequenceParallel]:
    world_size = group.size()
    try:
        obj = CACHE[id(group), dtype]
    except KeyError:
        if int(os.environ.get('DISABLE_FUSED_SEQUENCE_PARALLEL', '0')):
            obj = None
        elif world_size == 1:
            obj = None
        elif not _can_ranks_communicate_all_to_all_over_nvlink(group):
            obj = None
        else:
            obj = _FusedSequenceParallel(device, dtype, group, num_stripes)
        CACHE[id(group), dtype] = obj
    return obj