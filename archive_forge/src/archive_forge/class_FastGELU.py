import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm._C import ops
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.utils import set_weight_attrs
class FastGELU(nn.Module):

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        ops.gelu_fast(out, x)
        return out