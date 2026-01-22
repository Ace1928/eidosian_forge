import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set
from torch.ao.quantization.quantization_mappings import (
class ShadowLogger(Logger):
    """Class used in Shadow module to record the outputs of the original and
    shadow modules.
    """

    def __init__(self):
        super().__init__()
        self.stats['float'] = []
        self.stats['quantized'] = []

    def forward(self, x, y):
        """
        """
        if len(x) > 1:
            x = x[0]
        if len(y) > 1:
            y = y[0]
        self.stats['quantized'].append(x.detach())
        self.stats['float'].append(y.detach())