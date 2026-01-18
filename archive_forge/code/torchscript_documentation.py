from io import BytesIO
from typing import Any, Optional
import srsly
from ..compat import torch
from ..util import get_torch_default_device
from .pytorch import PyTorchShim
from .pytorch_grad_scaler import PyTorchGradScaler
A Thinc shim that wraps a TorchScript module.

    model:
        The TorchScript module. A value of `None` is also possible to
        construct a shim to deserialize into.
    mixed_precision:
        Enable mixed-precision. This changes whitelisted ops to run
        in half precision for better performance and lower memory use.
    grad_scaler:
        The gradient scaler to use for mixed-precision training. If this
        argument is set to "None" and mixed precision is enabled, a gradient
        scaler with the default configuration is used.
    device:
        The PyTorch device to run the model on. When this argument is
        set to "None", the default device for the currently active Thinc
        ops is used.
    