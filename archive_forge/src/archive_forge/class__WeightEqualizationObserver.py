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
class _WeightEqualizationObserver(nn.Module):
    """Observer for tracking the running min/max values of weight columns and
    rows, and computing the quantization parameters for the weight rows.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    This observer is made up of 1 PerChannelMinMaxObserver `weight_col_obs` used
    to record the running minimum and maximum of columns of incoming weight
    tensors. This observer is intended to be used along with an
    InputEqualizationObserver to calculate the equalization scale.

    The running minimum/maximum :math:`w_\\text{min/max}` are computed in the
    same way as :class:`~torch.ao.quantization.observer.PerChannelMinMaxObserver`.
    """

    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine, quant_min=None, quant_max=None, factory_kwargs=None) -> None:
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.ch_axis = 1
        per_channel_qscheme = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            per_channel_qscheme = qsheme_mapping_per_tensor_to_per_channel[qscheme]
        self.weight_col_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=dtype, qscheme=per_channel_qscheme, quant_min=quant_min, quant_max=quant_max, factory_kwargs=factory_kwargs)
        self.equalization_scale = torch.tensor(1)

    def forward(self, w_orig):
        if not (w_orig.ndim >= 2 and w_orig.ndim <= 5):
            raise ValueError('InputEqualizationObserver only supports Linear and Conv layers')
        return self.weight_col_obs(w_orig)

    def get_weight_col_minmax(self):
        return (self.weight_col_obs.min_val, self.weight_col_obs.max_val)

    def set_equalization_scale(self, equalization_scale):
        self.equalization_scale = equalization_scale
    with_args = classmethod(_with_args)