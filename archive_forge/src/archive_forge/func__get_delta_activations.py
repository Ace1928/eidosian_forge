import math
from typing import Any, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lycoris_utils import LycorisLayer
def _get_delta_activations(self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    delta_weight = self.get_delta_weight(adapter_name)
    base_layer = self.get_base_layer()
    return F.conv2d(input, delta_weight, stride=base_layer.stride, padding=base_layer.padding, dilation=base_layer.dilation, groups=base_layer.groups)