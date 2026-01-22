from typing import Dict, Iterable, List, Union, cast
from ..compat import has_torch_amp, torch
from ..util import is_torch_array
def _scale_tensor(self, tensor: 'torch.Tensor', scale_per_device: Dict['torch.device', 'torch.Tensor'], inplace: bool):
    if not has_torch_amp:
        raise ValueError('Gradient scaling is not supported, requires capable GPU and torch>=1.9.0')
    if not tensor.is_cuda:
        msg = 'Gradient scaling is only supported for CUDA tensors. If you are using PyTorch models, you can avoid this error by disabling mixed-precision support.'
        raise ValueError(msg)
    device = tensor.device
    if device not in scale_per_device:
        scale_per_device[device] = self._scale.to(device=device)
    scale = scale_per_device[device]
    if inplace:
        return tensor.mul_(scale)
    else:
        return tensor * scale