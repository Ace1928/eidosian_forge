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
@no_type_check
def _get_orig_buffer_dtypes(state: _FSDPState, buffer_names: List[str]) -> List[torch.dtype]:
    """
    Returns the original buffer types of the given buffer names.
    """
    buffer_dtypes: List[torch.dtype] = []
    for buffer_name in buffer_names:
        _p_assert(buffer_name in state._buffer_name_to_orig_dtype, f'{buffer_name} is missing from pre-computed dict on rank {state.rank}, which only has keys {state._buffer_name_to_orig_dtype.keys()}')
        buffer_dtypes.append(state._buffer_name_to_orig_dtype[buffer_name])
    return buffer_dtypes