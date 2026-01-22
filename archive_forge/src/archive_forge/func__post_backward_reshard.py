import functools
import logging
from enum import auto, Enum
from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.graph import register_multi_grad_hook
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._flat_param import (
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.utils import (
from torch.utils import _pytree as pytree
def _post_backward_reshard(state: _FSDPState, handle: FlatParamHandle, *unused: Any) -> None:
    free_unsharded_flat_param = _should_free_in_backward(state, handle)
    _reshard(state, handle, free_unsharded_flat_param)
    with torch.profiler.record_function('FullyShardedDataParallel._post_backward_prefetch'):
        _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)