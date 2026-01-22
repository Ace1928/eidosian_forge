import contextlib
import logging
import math
import warnings
from typing import Any, Callable, cast, Dict, Generator, Iterator, no_type_check, Tuple
import torch
import torch.distributed as dist
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp.api import (
from torch.distributed.utils import _replace_by_prefix
from ._fsdp_extensions import (
from ._unshard_param_utils import _unshard_fsdp_state_params, FLAT_PARAM
@no_type_check
def _full_post_state_dict_hook(module: nn.Module, fsdp_state: _FSDPState, state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Hook that runs after model.state_dict() is called before returning result to
    user. For FSDP, we may have to clone the tensors in state_dict as params go
    back to sharded version after _unshard_fsdp_state_params ends, and also remove
    the ``FSDP_WRAPPED_MODULE`` prefix.
    """

    def param_hook(state_dict: Dict[str, Any], prefix: str, fqn: str) -> None:
        clean_key = fqn
        clean_prefix = clean_tensor_name(prefix)
        if clean_key.startswith(clean_prefix):
            clean_key = clean_key[len(clean_prefix):]
        if not getattr(state_dict[fqn], '_has_been_cloned', False):
            try:
                state_dict[fqn] = state_dict[fqn].clone().detach()
                state_dict[fqn]._has_been_cloned = True
            except BaseException as e:
                warnings.warn(f'Failed to clone() tensor with name {fqn} on rank {fsdp_state.rank}. This may mean that this state_dict entry could point to invalid memory regions after returning from state_dict() call if this parameter is managed by FSDP. Please check clone implementation of {fqn}. Error: {str(e)}')
    return _common_unshard_post_state_dict_hook(module, fsdp_state, state_dict, prefix, param_hook)