from typing import Any, Callable, Dict, Optional, Tuple, cast
from ..compat import torch
from ..config import registry
from ..model import Model
from ..shims import PyTorchGradScaler, PyTorchShim
from ..types import ArgsKwargs, Floats3d, Padded
from ..util import (
Return the output of the wrapped PyTorch model for the given input,
    along with a callback to handle the backward pass.
    