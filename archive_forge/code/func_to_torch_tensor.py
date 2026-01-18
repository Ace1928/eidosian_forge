import functools
import importlib
import logging
import os
import tempfile
import torch
from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend
def to_torch_tensor(nd_tensor):
    """A helper function to transfer a NDArray to torch.tensor."""
    if nd_tensor.dtype == 'bool':
        return torch.from_numpy(nd_tensor.numpy())
    return torch.utils.dlpack.from_dlpack(nd_tensor.to_dlpack())