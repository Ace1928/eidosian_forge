from typing import List
import torch
from torch import Tensor
from torch._ops import ops
Operation equivalent to ``torch.ops.quantized.matmul(Tensor, Tensor)``