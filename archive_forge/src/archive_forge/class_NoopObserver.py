import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class NoopObserver(ObserverBase):
    """
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Primarily used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: Quantized data type
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
    """

    def __init__(self, dtype=torch.float16, custom_op_name='') -> None:
        super().__init__(dtype=dtype, is_dynamic=False)
        self.dtype = dtype
        self.custom_op = custom_op_name

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception('calculate_qparams should not be called for NoopObserver')