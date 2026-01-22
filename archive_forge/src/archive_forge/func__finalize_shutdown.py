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
def _finalize_shutdown():
    try:
        _destroy_rref_context(_ignore_rref_leak)
    finally:
        _get_current_rpc_agent().shutdown()
        _cleanup_python_rpc_handler()
        _reset_current_rpc_agent()