import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from itertools import count
from typing import (
from weakref import ReferenceType
import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
def _pt2_selective_checkpoint_context_fn_gen(policy_fn):
    """
    A helper function that generates a pair of contexts to be later passed into
    `torch.utils.checkpoint` API to implment selective checkpointing.

    .. warning::
        This is context_fn is intended for use with torch.compile only.

    Args:
        policy_fn (Callable[[Callable, List[Any], Dict[str, Any]], bool]): Policy function
            to decide whether a particular op should be recomputed in backward pass or not.
            In eager mode:
                If policy_fn(...) returns True, the op is guaranteed to NOT be recomputed.
                If policy_fn(...) returns False, the op is guaranteed to be recomputed.
            In torch.compile mode:
                If policy_fn(...) returns True, the op is guaranteed to NOT be recomputed.
                If policy_fn(...) returns False, the op may or may not be recomputed
                (it's up to the partitioner to decide).

    Returns:
        A pair of generated contexts.

    Example:
        >>> # xdoctest: +REQUIRES(LINUX)
        >>>
        >>> def get_custom_policy():
        >>>     no_recompute_list = [
        >>>         torch.ops.aten.mm.default,
        >>>     ]
        >>>     def custom_policy(mode, func, *args, **kwargs):
        >>>         return func in no_recompute_list
        >>>     return custom_policy
        >>>
        >>> def selective_checkpointing_context_fn():
        >>>     return _pt2_selective_checkpoint_context_fn_gen(get_custom_policy())
        >>>
        >>> def gn(x, y):
        >>>     return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
        >>>
        >>> def fn(x, y):
        >>>     return torch.utils.checkpoint.checkpoint(
        >>>         gn, x, y,
        >>>         use_reentrant=False,
        >>>         context_fn=selective_checkpointing_context_fn,
        >>>     )
        >>>
        >>> x = torch.randn(4, 4, requires_grad=True)
        >>> y = torch.randn(4, 4, requires_grad=True)
        >>>
        >>> compiled_fn = torch.compile(fn)
    """
    storage: Dict[Any, List[Any]] = defaultdict(list)
    return (_CachingTorchDispatchMode(policy_fn, storage), _CachedTorchDispatchMode(policy_fn, storage))