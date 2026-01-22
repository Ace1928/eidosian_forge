import torch
from typing import Iterable, Optional
def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    """Check if the parameters are located on the same device.

    Currently, the conversion between model parameters and single vector form is not supported
    for multiple allocations, e.g. parameters in different GPUs/PrivateUse1s, or mixture of CPU/GPU/PrivateUse1.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """
    support_device_types = ['cuda', torch._C._get_privateuse1_backend_name()]
    if old_param_device is None:
        old_param_device = param.get_device() if param.device.type in support_device_types else -1
    else:
        warn = False
        if param.device.type in support_device_types:
            warn = param.get_device() != old_param_device
        else:
            warn = old_param_device != -1
        if warn:
            raise TypeError('Found two parameters on different devices, this is currently not supported.')
    return old_param_device