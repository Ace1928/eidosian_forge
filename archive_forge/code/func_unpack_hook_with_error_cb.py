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
def unpack_hook_with_error_cb(holder):
    try:
        return unpack_hook(holder)
    except CheckpointError as e:
        frame.unpack_error_cb(e)