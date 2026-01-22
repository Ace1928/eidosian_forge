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
def _is_compiling(func, args, kwargs):
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if isinstance(arg, torch._subclasses.functional_tensor.FunctionalTensor):
                arg = torch._from_functional_tensor(arg.elem)
            if isinstance(arg, torch._subclasses.FakeTensor):
                return True
    return False