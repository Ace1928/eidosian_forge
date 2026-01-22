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
def _handle_compile_in_recompute_ctx(self, should_not_recompute, func, args, kwargs):
    if func in _ignored_ops:
        return func(*args, **kwargs)
    out = self.pop_from_storage(func, args, kwargs)
    return out