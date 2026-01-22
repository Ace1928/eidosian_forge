import collections
import contextlib
import functools
import inspect
import logging
import threading
from typing import Dict, Generic, TypeVar, Set, Any, TYPE_CHECKING
import torch
from torch.futures import Future
from torch._C._distributed_rpc import (
from .internal import (
from .constants import DEFAULT_SHUTDOWN_TIMEOUT, UNSET_RPC_TIMEOUT
from ._utils import _group_membership_management, _update_group_membership
def _enable_rpc_profiler(should_profile, qualified_name, func, rpc_type, dst_worker_info):
    ctx_manager = contextlib.nullcontext()
    if should_profile:
        if qualified_name is None:
            func_name = torch._jit_internal._qualified_name(func) if isinstance(func, torch.jit.ScriptFunction) else func.__qualname__
        else:
            func_name = qualified_name
        rpc_profiling_key = _build_rpc_profiling_key(rpc_type, func_name, get_worker_info().name, dst_worker_info.name)
        RemoteProfilerManager.set_current_profiling_key(rpc_profiling_key)
        ctx_manager = torch.autograd.profiler.record_function(rpc_profiling_key)
    return ctx_manager