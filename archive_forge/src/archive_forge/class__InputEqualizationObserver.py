import warnings
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from ..observer import _with_args, ObserverBase, PerChannelMinMaxObserver
from ..utils import _parent_name, check_min_max_valid
from .utils import (
class _InputEqualizationObserver(nn.Module):
    """Observer for tracking the running min/max values of input columns, and
    computing the quantization parameters for the overall min/max input values.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    The running minimum/maximum :math:`x_\\text{min/max}` are computed in the
    same way as :class:`~torch.ao.quantization.observer.PerChannelMinMaxObserver`,
    with the difference that the running min/max values are stored per column.
    This observer is intended to be used along with a WeightEqualizationObserver
    to calculate the equalization scale.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, quant_min=None, quant_max=None, factory_kwargs=None) -> None:
        super().__init__()
        if qscheme not in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            raise TypeError('Input qscheme must be per-tensor')
        self.dtype = dtype
        self.qscheme = qscheme
        per_channel_qscheme = qsheme_mapping_per_tensor_to_per_channel[qscheme]
        self.input_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=dtype, qscheme=per_channel_qscheme, quant_min=quant_min, quant_max=quant_max, factory_kwargs=factory_kwargs)
        self.equalization_scale = torch.tensor(1)
        self.equalization_shape: List[int] = []

    def forward(self, x_orig):
        if not (x_orig.ndim >= 2 and x_orig.ndim <= 5):
            raise ValueError('InputEqualizationObserver only supports Linear and Conv layers')
        self.equalization_shape = [1] * x_orig.ndim
        self.equalization_shape[1] = x_orig.size(1)
        return self.input_obs(x_orig)

    def get_input_minmax(self):
        return (self.input_obs.min_val, self.input_obs.max_val)

    def set_equalization_scale(self, equalization_scale):
        if equalization_scale.nelement() == 1 and equalization_scale == torch.tensor(1):
            return
        self.equalization_scale = torch.reshape(equalization_scale, self.equalization_shape)

    def calculate_scaled_minmax(self):
        """ Returns the scaled min/max inputs
        """
        if self.equalization_scale.nelement() == 1 and self.equalization_scale == torch.tensor(1):
            warnings.warn('Must call calculate_equalization_scale before calling calculate_scaled_minmax. ' + 'Will not scale the next quantization observer.')
            return (None, None)
        min_inputs, max_inputs = self.get_input_minmax()
        equalization_scale_reshaped = reshape_scale(self.equalization_scale, 0, min_inputs)
        min_input_scaled = torch.min(torch.mul(min_inputs, equalization_scale_reshaped))
        max_input_scaled = torch.max(torch.mul(max_inputs, equalization_scale_reshaped))
        return (min_input_scaled, max_input_scaled)
    with_args = classmethod(_with_args)