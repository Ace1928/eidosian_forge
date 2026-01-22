import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import torch
from torchvision import tv_tensors
def _register_five_ten_crop_kernel_internal(functional, input_type):
    registry = _KERNEL_REGISTRY.setdefault(functional, {})
    if input_type in registry:
        raise TypeError(f"Functional '{functional}' already has a kernel registered for type '{input_type}'.")

    def wrap(kernel):

        @functools.wraps(kernel)
        def wrapper(inpt, *args, **kwargs):
            output = kernel(inpt, *args, **kwargs)
            container_type = type(output)
            return container_type((tv_tensors.wrap(o, like=inpt) for o in output))
        return wrapper

    def decorator(kernel):
        registry[input_type] = wrap(kernel) if issubclass(input_type, tv_tensors.TVTensor) else kernel
        return kernel
    return decorator