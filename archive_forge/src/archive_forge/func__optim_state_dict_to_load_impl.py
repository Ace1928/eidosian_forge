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
def _optim_state_dict_to_load_impl(optim_state_dict: Dict[str, Any], model: torch.nn.Module, optim_input: Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]=None, optim: Optional[torch.optim.Optimizer]=None, full_state_dict: bool=True, rank0_only: bool=False, is_named_optimizer: bool=False, group: Optional[dist.ProcessGroup]=None) -> Dict[str, Any]:
    """
        Convert an optimizer state-dict so that it can be loaded into the optimizer associated with the FSDP model.

        This is the internal API that is used by all the load optim_state_dict implementations.
        Given model, optim, and the saved optim_state_dict, this API adds the FSDP
        internal information and internal sharding to the optim_state_dict.
        """
    if full_state_dict:
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(optim_input, optim)
    else:
        using_optim_input = False
        assert optim_input is None and (not rank0_only)
    use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[0]._use_orig_params
    assert all((use_orig_params == m._use_orig_params for m in FullyShardedDataParallel.fsdp_modules(model))), 'Not all FSDP modules have the same _use_orig_params value'
    if rank0_only and dist.get_rank(group) > 0:
        optim_state_dict = {}
    sharded_osd = _flatten_optim_state_dict(optim_state_dict, model=model, use_orig_params=use_orig_params, optim=optim if is_named_optimizer else None, rank0_only=rank0_only, group=group)
    return _rekey_sharded_optim_state_dict(sharded_osd, model=model, optim=optim, optim_input=optim_input, using_optim_input=using_optim_input, is_named_optimizer=is_named_optimizer)