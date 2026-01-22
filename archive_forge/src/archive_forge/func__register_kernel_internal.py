import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import torch
from torchvision import tv_tensors
def _register_kernel_internal(functional, input_type, *, tv_tensor_wrapper=True):
    registry = _KERNEL_REGISTRY.setdefault(functional, {})
    if input_type in registry:
        raise ValueError(f'Functional {functional} already has a kernel registered for type {input_type}.')

    def decorator(kernel):
        registry[input_type] = _kernel_tv_tensor_wrapper(kernel) if issubclass(input_type, tv_tensors.TVTensor) and tv_tensor_wrapper else kernel
        return kernel
    return decorator