from typing import List
import torch
from torch import Tensor
from torch._ops import ops
def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
    r = ops.quantized.add_relu(x, y, scale=self.scale, zero_point=self.zero_point)
    r = self.activation_post_process(r)
    return r