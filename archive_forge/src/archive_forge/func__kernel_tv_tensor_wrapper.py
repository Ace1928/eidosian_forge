import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import torch
from torchvision import tv_tensors
def _kernel_tv_tensor_wrapper(kernel):

    @functools.wraps(kernel)
    def wrapper(inpt, *args, **kwargs):
        output = kernel(inpt.as_subclass(torch.Tensor), *args, **kwargs)
        return tv_tensors.wrap(output, like=inpt)
    return wrapper