import inspect
from copy import deepcopy
from functools import wraps
from typing import (
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch import nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override
from lightning_fabric.plugins import Precision
from lightning_fabric.strategies import Strategy
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.data import _set_sampler_epoch
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import Optimizable
def _capture_compile_kwargs(compile_fn: Callable) -> Callable:
    """Wraps the ``torch.compile`` function and captures the compile arguments.

    We extract the compile arguments so that we can reapply ``torch.compile`` in ``Fabric.setup()`` with the
    same arguments as the user passed to the original call. The arguments get stored in a dictionary
    ``_compile_kwargs`` on the returned compiled module.

    """

    @wraps(compile_fn)
    def _capture(*args: Any, **kwargs: Any) -> Any:
        if not args or not isinstance(args[0], nn.Module):
            return compile_fn(*args, **kwargs)
        model = args[0]
        compiled_model = compile_fn(model, **kwargs)
        compiled_model._compile_kwargs = deepcopy(kwargs)
        return compiled_model
    return _capture