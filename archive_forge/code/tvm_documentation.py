import functools
import importlib
import logging
import os
import tempfile
import torch
from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend
A helper function to transfer a torch.tensor to NDArray.