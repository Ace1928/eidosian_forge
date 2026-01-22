import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class ReuseInputObserver(ObserverBase):
    """ This observer is used when we want to reuse the observer from the operator
    that produces the input Tensor, typically used for operators like reshape, e.g.
    ```
    x0 = ...
    x1 = x0.reshape()
    ```
    if we configure x0 to be observed by some observer, let's say MinMaxObserver,
    and reshape is configured with ReuseInputObserver, we'll reuse the observer instance
    for x0 for x1 (output of reshape). If x0 is not observed, we also won't observe x1.

    Note: this is only enabled in FX Graph Mode Quantization
    """

    def __init__(self):
        super().__init__(torch.quint8, is_dynamic=False)

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception('calculate_qparams should not be called for ReuseInputObserver')