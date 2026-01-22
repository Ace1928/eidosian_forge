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
class _NoopSaveInputs(torch.autograd.Function):

    @staticmethod
    def forward(*args):
        return torch.empty((0,))

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        tensor_indices, tensors = zip(*[(i, o) for i, o in enumerate(inputs) if isinstance(o, torch.Tensor)])
        idx2saved_idx = {b: a for a, b in enumerate(tensor_indices)}
        args = [None if isinstance(o, torch.Tensor) else o for o in inputs]

        def get_args(saved_tensors):
            ret = [saved_tensors[idx2saved_idx[i]] if i in tensor_indices else o for i, o in enumerate(args)]
            return ret[1:]
        ctx.get_args = get_args
        ctx.save_for_backward(*tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise AssertionError('Did not expect to backward on this graph')