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
def _root_pre_forward(state: _FSDPState, module: nn.Module, args, kwargs) -> None:
    """
    Runs pre-forward logic specific to the root FSDP instance, which should run
    before any individual module's pre-forward. This starts with an attempt at
    lazy initialization (which only runs non-vacuously once). Otherwise, if
    this is called on a non-root FSDP instance, then it returns directly.

    Args:
        module (nn.Module): Module for which this logic tries to run. It may or
            may not be the root. If not, then this method does not do anything.
    """
    with torch.profiler.record_function('FullyShardedDataParallel._root_pre_forward'):
        _lazy_init(state, module)
        _p_assert(state._is_root is not None, 'Expects a root FSDP to have been set')
        if not state._is_root:
            if _is_composable(state):
                return _root_cast_forward_input(state, module, args, kwargs)
            return (args, kwargs)
        handle = state._handle
        if handle:
            should_cast_buffers_to_full_prec = handle._force_full_precision
        else:
            should_cast_buffers_to_full_prec = True
        if should_cast_buffers_to_full_prec:
            _cast_buffers_to_dtype_and_device(buffers=dict(module.named_buffers()).values(), buffer_dtypes=list(state._buffer_name_to_orig_dtype.values()), device=state.compute_device)
            state._needs_buffer_dtype_restore_check = True
        elif getattr(state, '_needs_buffer_dtype_restore_check', False):
            buffers, buffer_dtypes_for_computation = _get_buffers_and_dtypes_for_computation(state, module)
            if len(buffers) > 0 and len(buffer_dtypes_for_computation) > 0:
                if any((buffer.dtype != buffer_dtype_for_computation for buffer, buffer_dtype_for_computation in zip(buffers, buffer_dtypes_for_computation))):
                    _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes_for_computation, state.compute_device)
            state._needs_buffer_dtype_restore_check = False
        if state.forward_prefetch:
            handles = []
            for fsdp_state in state._all_fsdp_states:
                if fsdp_state._handle:
                    handles.append(fsdp_state._handle)
            for handle in handles:
                handle._needs_pre_forward_unshard = True
                handle._prefetched = False
        _wait_for_computation_stream(state._device_handle.current_stream(), state._unshard_stream, state._pre_unshard_stream)
        _reset_flat_param_grad_info_if_needed(state._all_handles)
        with torch.profiler.record_function('FullyShardedDataParallel._to_kwargs'):
            args_tuple, kwargs_tuple = _to_kwargs(args, kwargs, state.compute_device, False)
        args = args_tuple[0]
        kwargs = kwargs_tuple[0]
        return _root_cast_forward_input(state, module, args, kwargs)