import os
import sys
import warnings
from contextlib import contextmanager
import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule
class CudnnModule(PropModule):

    def __init__(self, m, name):
        super().__init__(m, name)
    enabled = ContextProp(torch._C._get_cudnn_enabled, torch._C._set_cudnn_enabled)
    deterministic = ContextProp(torch._C._get_cudnn_deterministic, torch._C._set_cudnn_deterministic)
    benchmark = ContextProp(torch._C._get_cudnn_benchmark, torch._C._set_cudnn_benchmark)
    benchmark_limit = None
    if is_available():
        benchmark_limit = ContextProp(torch._C._cuda_get_cudnn_benchmark_limit, torch._C._cuda_set_cudnn_benchmark_limit)
    allow_tf32 = ContextProp(torch._C._get_cudnn_allow_tf32, torch._C._set_cudnn_allow_tf32)