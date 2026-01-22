from enum import Enum
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers import _is_triton_available
from collections import namedtuple
class PostNorm(nn.Module, RequiresWrappedInputs):
    """Adds LayerNorm after computing attention"""

    def __init__(self, d_norm: int, sublayer: nn.Module, normalization: NormalizationType, use_triton: bool=True):
        super().__init__()
        if _is_triton_available() and use_triton and (normalization == NormalizationType.LayerNorm):
            self.norm: Union[nn.LayerNorm, FusedLayerNorm] = FusedLayerNorm(d_norm)
        else:
            self.norm = get_normalization_layer(normalization)(d_norm)
        self.sublayer = sublayer
        self.wrap_inputs = isinstance(sublayer, RequiresWrappedInputs)

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        if self.wrap_inputs:
            x = self.sublayer(inputs=inputs, **kwargs)
        else:
            x = self.sublayer(*inputs, **kwargs)
        return self.norm(x)