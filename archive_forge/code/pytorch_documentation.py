import contextlib
import itertools
from io import BytesIO
from typing import Any, Callable, Dict, Optional, cast
import srsly
from ..backends import CupyOps, context_pools, get_current_ops, set_gpu_allocator
from ..compat import torch
from ..optimizers import Optimizer
from ..types import ArgsKwargs, FloatsXd
from ..util import (
from .pytorch_grad_scaler import PyTorchGradScaler
from .shim import Shim
Pass the inputs through to the underlying PyTorch model, keeping
        track of which items in the input are tensors requiring gradients.
        If the model returns a single value, it is converted into a one-element tuple.
        Return the outputs and a callback to backpropagate.
        