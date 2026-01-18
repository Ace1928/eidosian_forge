from typing import Any, Callable, Optional
from ..compat import torch
from ..model import Model
from ..shims import PyTorchGradScaler, PyTorchShim, TorchScriptShim
from .pytorchwrapper import (
Convert a PyTorch wrapper to a TorchScript wrapper. The embedded PyTorch
    `Module` is converted to `ScriptModule`.
    