import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class MinMaxObserver(UniformQuantizationObserverBase):
    """Observer module for computing the quantization parameters based on the
    running min and max values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    Given running min/max as :math:`x_\\text{min}` and :math:`x_\\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:

    The running minimum/maximum :math:`x_\\text{min/max}` is computed as:

    .. math::

        \\begin{array}{ll}
        x_\\text{min} &= \\begin{cases}
            \\min(X) & \\text{if~}x_\\text{min} = \\text{None} \\\\
            \\min\\left(x_\\text{min}, \\min(X)\\right) & \\text{otherwise}
        \\end{cases}\\\\
        x_\\text{max} &= \\begin{cases}
            \\max(X) & \\text{if~}x_\\text{max} = \\text{None} \\\\
            \\max\\left(x_\\text{max}, \\max(X)\\right) & \\text{otherwise}
        \\end{cases}\\\\
        \\end{array}

    where :math:`X` is the observed tensor.

    The scale :math:`s` and zero point :math:`z` are then computed as:

    .. math::

        \\begin{aligned}
            \\text{if Symmetric:}&\\\\
            &s = 2 \\max(|x_\\text{min}|, x_\\text{max}) /
                \\left( Q_\\text{max} - Q_\\text{min} \\right) \\\\
            &z = \\begin{cases}
                0 & \\text{if dtype is qint8} \\\\
                128 & \\text{otherwise}
            \\end{cases}\\\\
            \\text{Otherwise:}&\\\\
                &s = \\left( x_\\text{max} - x_\\text{min}  \\right ) /
                    \\left( Q_\\text{max} - Q_\\text{min} \\right ) \\\\
                &z = Q_\\text{min} - \\text{round}(x_\\text{min} / s)
        \\end{aligned}

    where :math:`Q_\\text{min}` and :math:`Q_\\text{max}` are the minimum and
    maximum of the quantized data type.

    .. warning:: :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False, quant_min=None, quant_max=None, factory_kwargs=None, eps=torch.finfo(torch.float32).eps, is_dynamic=False, **kwargs) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError("MinMaxObserver's qscheme only support torch.per_tensor_symmetric                     and torch.per_tensor_affine.")
        super().__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max, factory_kwargs=factory_kwargs, eps=eps, is_dynamic=is_dynamic, **kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer('min_val', torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val', torch.tensor(float('-inf'), **factory_kwargs))
        if self.qscheme == torch.per_tensor_symmetric and self.reduce_range and (self.dtype == torch.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric                                        quantization for quint8')

    def forward(self, x_orig):
        """Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return f'min_val={self.min_val}, max_val={self.max_val}'

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float('inf')))
        self.max_val.copy_(torch.tensor(float('-inf')))