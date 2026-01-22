import contextlib
from typing import Union
import torch
from torch._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend
class cuBLASModule:

    def __getattr__(self, name):
        if name == 'allow_tf32':
            return torch._C._get_cublas_allow_tf32()
        elif name == 'allow_fp16_reduced_precision_reduction':
            return torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
        elif name == 'allow_bf16_reduced_precision_reduction':
            return torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
        raise AttributeError('Unknown attribute ' + name)

    def __setattr__(self, name, value):
        if name == 'allow_tf32':
            return torch._C._set_cublas_allow_tf32(value)
        elif name == 'allow_fp16_reduced_precision_reduction':
            return torch._C._set_cublas_allow_fp16_reduced_precision_reduction(value)
        elif name == 'allow_bf16_reduced_precision_reduction':
            return torch._C._set_cublas_allow_bf16_reduced_precision_reduction(value)
        raise AttributeError('Unknown attribute ' + name)