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
def _get_should_profile():
    ActiveProfilerType = torch._C._profiler.ActiveProfilerType
    return torch.autograd._profiler_enabled() and torch._C._autograd._profiler_type() == ActiveProfilerType.LEGACY