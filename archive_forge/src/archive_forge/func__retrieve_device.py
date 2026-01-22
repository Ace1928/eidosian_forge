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
def _retrieve_device(module: nn.Module) -> torch.device:
    """Validates all parameters in the Module have the same device and returns
    the appropriate device.

    Args:
        An ``nn.Module`` to process.

    Returns:
        ``torch.Device`` for the entire module.

    Raises:
        ValueError:
            If devices for ``nn.Module`` parameters are not all same.
    """
    device = None
    for parameter in module.parameters():
        if device is None:
            device = parameter.device
        elif device != parameter.device:
            raise ValueError(f'nn.Module: {module}, should have all parameters on a single device, please use .to() to place the module on a single device')
    return device if device is not None else torch.device('cpu')