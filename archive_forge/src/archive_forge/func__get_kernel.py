import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import torch
from torchvision import tv_tensors
def _get_kernel(functional, input_type, *, allow_passthrough=False):
    registry = _KERNEL_REGISTRY.get(functional)
    if not registry:
        raise ValueError(f'No kernel registered for functional {functional.__name__}.')
    for cls in input_type.__mro__:
        if cls in registry:
            return registry[cls]
        elif cls is tv_tensors.TVTensor:
            break
    if allow_passthrough:
        return lambda inpt, *args, **kwargs: inpt
    raise TypeError(f'Functional F.{functional.__name__} supports inputs of type {registry.keys()}, but got {input_type} instead.')