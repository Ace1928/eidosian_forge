import functools
import torch
import torch.distributed as dist
from enum import Enum
def _quantize_tensor_list(tensor_list, qtype):
    if not isinstance(tensor_list, list) or not all((isinstance(p, torch.Tensor) for p in tensor_list)):
        raise RuntimeError(f'_quantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}')
    quantized_tensor_list = [_quantize_tensor(t, qtype) for t in tensor_list]
    return quantized_tensor_list