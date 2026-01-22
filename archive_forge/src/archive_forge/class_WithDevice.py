from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast
import torch
from torch import Tensor, nn
from torch.distributed.rpc import RRef
import torch.autograd
import torch.cuda
from . import microbatch
from .batchnorm import DeferredBatchNorm
from .pipeline import Pipeline
from .skip.layout import inspect_skip_layout
from .skip.skippable import verify_skippables
from .stream import AbstractStream, new_stream
class WithDevice(nn.Module):
    """
    Wraps an ``nn.Module`` which is part of ``nn.Sequential`` passed into :class:`Pipe`
    that overrides the device for that module. In cases where :class:`Pipe`
    can't implicitly determine the device for the module and places it on CPU,
    this wrapper can be used to override the implicit behavior and explicitly
    specify which device a module should run on.

    The provided module is also moved to the given device via ``.to(device)``
    by :class:`Pipe`

    Args:
        module(:class:`torch.nn.Module`): The module to be wrapped.
        device(:class:`torch.device`): The device to run the module on.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> fc1 = nn.Linear(16, 8).cuda(0)
        >>> fc2 = nn.Linear(8, 4).cuda(1)
        >>> dropout = nn.Dropout()
        >>>
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
        >>> # Dropout does not have any parameters/buffers, but we want to
        >>> # run it on cuda:1 to avoid any GPU to CPU transfers.
        >>> model = nn.Sequential(fc1, fc2, WithDevice(dropout, 'cuda:1'))
        >>> # xdoctest: +SKIP("Needs RPC framework init")
        >>> model = Pipe(model, chunks=8)
    """

    def __init__(self, module: nn.Module, device: torch.device):
        super().__init__()
        self._module = module
        self._device = torch.device(device)

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    @property
    def module(self):
        return self._module

    @property
    def device(self):
        return self._device