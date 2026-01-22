import contextlib
import copy
import functools
import math
import traceback
import warnings
from contextlib import contextmanager
from enum import auto, Enum
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParameter
from ._optim_utils import (
from ._state_dict_utils import _register_all_state_dict_hooks
from ._unshard_param_utils import (
from .wrap import CustomPolicy, ModuleWrapPolicy
@staticmethod
def _optim_state_dict_impl(model: torch.nn.Module, optim: torch.optim.Optimizer, optim_state_dict: Dict[str, Any], optim_input: Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]=None, rank0_only: bool=True, full_state_dict: bool=True, group: Optional[dist.ProcessGroup]=None, cpu_offload: bool=True) -> Dict[str, Any]:
    """Transform the state-dict of an optimizer corresponding to a sharded model.

        This is the internal API that is used by all the optim_state_dict implementations.
        Given model, optim, the original optim_state_dict, this API removes the
        FSDP internal information and internal sharding from the optim_state_dict.
        """
    if full_state_dict:
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(optim_input, optim)
    else:
        using_optim_input = False
        assert optim_input is None and (not rank0_only)
    use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[0]._use_orig_params
    assert all((use_orig_params == m._use_orig_params for m in FullyShardedDataParallel.fsdp_modules(model))), 'Not all FSDP modules have the same _use_orig_params value'
    return _optim_state_dict(model=model, optim=optim, optim_state_dict=optim_state_dict, optim_input=optim_input, rank0_only=rank0_only, shard_state=not full_state_dict, group=group, using_optim_input=using_optim_input, use_orig_params=use_orig_params, cpu_offload=cpu_offload)