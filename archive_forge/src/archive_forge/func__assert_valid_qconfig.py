from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def _assert_valid_qconfig(qconfig: Optional[QConfig], mod: torch.nn.Module) -> None:
    """
    Verifies that this `qconfig` is valid.
    """
    if qconfig is None:
        return
    is_conv_transpose_mod = isinstance(mod, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d))
    if is_conv_transpose_mod:
        if qconfig.weight is None:
            return
        example_observer = qconfig.weight()
        is_per_channel = isinstance(example_observer, (torch.ao.quantization.PerChannelMinMaxObserver, torch.ao.quantization.MovingAveragePerChannelMinMaxObserver))
        assert not is_per_channel, 'Per channel weight observer is not supported yet for ConvTranspose{n}d.'