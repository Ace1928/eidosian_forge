import contextlib
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities.model_helpers import _ModuleMode
from pytorch_lightning.utilities.rank_zero import WarningCache
class LayerSummary:
    """Summary class for a single layer in a :class:`~pytorch_lightning.core.LightningModule`. It collects the
    following information:

    - Type of the layer (e.g. Linear, BatchNorm1d, ...)
    - Input shape
    - Output shape
    - Number of parameters

    The input and output shapes are only known after the example input array was
    passed through the model.

    Example::

        >>> model = torch.nn.Conv2d(3, 8, 3)
        >>> summary = LayerSummary(model)
        >>> summary.num_parameters
        224
        >>> summary.layer_type
        'Conv2d'
        >>> output = model(torch.rand(1, 3, 5, 5))
        >>> summary.in_size
        [1, 3, 5, 5]
        >>> summary.out_size
        [1, 8, 3, 3]

    Args:
        module: A module to summarize

    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module
        self._hook_handle = self._register_hook()
        self._in_size: Optional[Union[str, List]] = None
        self._out_size: Optional[Union[str, List]] = None

    def __del__(self) -> None:
        self.detach_hook()

    def _register_hook(self) -> Optional[RemovableHandle]:
        """Registers a hook on the module that computes the input- and output size(s) on the first forward pass. If the
        hook is called, it will remove itself from the from the module, meaning that recursive models will only record
        their input- and output shapes once. Registering hooks on :class:`~torch.jit.ScriptModule` is not supported.

        Return:
            A handle for the installed hook, or ``None`` if registering the hook is not possible.

        """

        def hook(_: nn.Module, inp: Any, out: Any) -> None:
            if len(inp) == 1:
                inp = inp[0]
            self._in_size = parse_batch_shape(inp)
            self._out_size = parse_batch_shape(out)
            assert self._hook_handle is not None
            self._hook_handle.remove()

        def hook_with_kwargs(_: nn.Module, args: Any, kwargs: Any, out: Any) -> None:
            inp = (*args, *kwargs.values()) if kwargs is not None else args
            hook(_, inp, out)
        handle = None
        if not isinstance(self._module, torch.jit.ScriptModule):
            if _TORCH_GREATER_EQUAL_2_0:
                handle = self._module.register_forward_hook(hook_with_kwargs, with_kwargs=True)
            else:
                handle = self._module.register_forward_hook(hook)
        return handle

    def detach_hook(self) -> None:
        """Removes the forward hook if it was not already removed in the forward pass.

        Will be called after the summary is created.

        """
        if self._hook_handle is not None:
            self._hook_handle.remove()

    @property
    def in_size(self) -> Union[str, List]:
        return self._in_size or UNKNOWN_SIZE

    @property
    def out_size(self) -> Union[str, List]:
        return self._out_size or UNKNOWN_SIZE

    @property
    def layer_type(self) -> str:
        """Returns the class name of the module."""
        return str(self._module.__class__.__name__)

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters in this module."""
        return sum((math.prod(p.shape) if not _is_lazy_weight_tensor(p) else 0 for p in self._module.parameters()))