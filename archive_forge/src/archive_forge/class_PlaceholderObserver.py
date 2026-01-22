import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class PlaceholderObserver(ObserverBase):
    """
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Can be used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        quant_min: minimum value in quantized domain (TODO: align behavior with other observers)
        quant_max: maximum value in quantized domain
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
        compute_dtype (deprecated): if set, marks the future quantize function to use
                       dynamic quantization instead of static quantization.
                       This field is deprecated, use `is_dynamic=True` instead.
        is_dynamic: if True, the `quantize` function in the reference model
                    representation taking stats from this observer instance will
                    use dynamic quantization.
    """

    def __init__(self, dtype=torch.float32, custom_op_name='', compute_dtype=None, quant_min=None, quant_max=None, qscheme=None, eps=None, is_dynamic=False) -> None:
        super().__init__(dtype=dtype, is_dynamic=is_dynamic)
        if qscheme is None:
            qscheme = torch.per_tensor_affine
        if eps is None:
            eps = torch.finfo(torch.float32).eps
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        self.custom_op = custom_op_name
        if compute_dtype:
            is_dynamic = True
            warnings.warn('Please use `is_dynamic` instead of `compute_dtype`.                     `compute_dtype` will be deprecated in a future release                     of PyTorch.')

    def forward(self, x):
        return x

    @torch.jit.export
    def extra_repr(self):
        return f'dtype={self.dtype}, is_dynamic={self.is_dynamic}'

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception('calculate_qparams should not be called for PlaceholderObserver')