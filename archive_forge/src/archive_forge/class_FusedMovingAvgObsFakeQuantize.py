import torch
from torch.nn import Module
from torch.ao.quantization.observer import (
import re
from abc import ABC, abstractmethod
from typing import Any, Tuple
class FusedMovingAvgObsFakeQuantize(FakeQuantize):
    """Define a fused module to observe the tensor.

    Fused module that is used to observe the input tensor (compute min/max), compute
    scale/zero_point and fake_quantize the tensor.
    This module uses calculation similar MovingAverageMinMaxObserver for the inputs,
    to compute the min/max values in order to compute the scale/zero_point.
    The qscheme input in the observer is used to differentiate between symmetric/affine
    quantization scheme.

    The output of this module is given by
    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale

    Similar to :class:`~torch.ao.quantization.FakeQuantize`, and accepts the same attributes as the
    base class.

    """

    def __init__(self, observer: Any=MovingAverageMinMaxObserver, quant_min: int=0, quant_max: int=255, **observer_kwargs: Any) -> None:
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        assert isinstance(self.activation_post_process, (MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver)), 'Fused observer+fake_quant module only works with MovingAverageMinMaxObserver'
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.long))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.long))
        self.is_symmetric_quant = _is_symmetric_quant(self.activation_post_process.qscheme)

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def extra_repr(self) -> str:
        return 'fake_quant_enabled={}, observer_enabled={}, scale={}, zero_point={}, dtype={}, quant_min={}, quant_max={}, qscheme={}, reduce_range={}'.format(self.fake_quant_enabled, self.observer_enabled, self.scale, self.zero_point, self.dtype, self.activation_post_process.quant_min, self.activation_post_process.quant_max, self.qscheme, self.activation_post_process.reduce_range)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.fused_moving_avg_obs_fake_quant(X, self.observer_enabled, self.fake_quant_enabled, self.activation_post_process.min_val, self.activation_post_process.max_val, self.scale, self.zero_point, self.activation_post_process.averaging_constant, self.activation_post_process.quant_min, self.activation_post_process.quant_max, self.ch_axis, self.is_per_channel, self.is_symmetric_quant)