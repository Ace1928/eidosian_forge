import contextlib
import warnings
from typing import cast, Generator
import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParamHandle
@contextlib.contextmanager
def _unshard_params_recurse(module: nn.Module, state: _FSDPState, recurse: bool, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    """
    This is a helper for :func:`_unshard_params` that recursively calls
    :func:`_unshard_fsdp_state_params` on FSDP states if ``recurse=True``.
    NOTE: This runs lazy initialization.
    """
    _validate_unshard_params_args(state, writeback, rank0_only, offload_to_cpu, with_grads)
    if recurse:
        with contextlib.ExitStack() as stack:
            for state, fsdp_module in zip(*traversal_utils._get_fsdp_states_with_modules(module)):
                stack.enter_context(_unshard_params_recurse(module=fsdp_module, state=state, recurse=False, writeback=writeback, rank0_only=rank0_only, offload_to_cpu=offload_to_cpu, with_grads=with_grads))
            yield
        return
    _lazy_init(state, module)
    if state.training_state == TrainingState.FORWARD_BACKWARD:
        raise AssertionError('Cannot manually unshard parameters during forward/backward')
    elif state.training_state == TrainingState.SUMMON_FULL_PARAMS:
        raise AssertionError('Cannot manually unshard parameters when already unsharding parameters')
    with _unshard_fsdp_state_params(module=module, state=state, writeback=writeback, rank0_only=rank0_only, offload_to_cpu=offload_to_cpu, with_grads=with_grads):
        try:
            state.training_state = TrainingState.SUMMON_FULL_PARAMS
            yield
        finally:
            state.training_state = TrainingState.IDLE