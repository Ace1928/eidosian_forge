import torch
import torch.distributed as dist
from torch import nn
def _dequantize_per_tensor_cuda(y, scale, zero_point):
    x = scale * (y.to(torch.float32) - zero_point)
    return x