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
@contextlib.contextmanager
def use_params(self, params):
    key_prefix = f'pytorch_{self.id}_'
    state_dict = {}
    for k, v in params.items():
        if hasattr(k, 'startswith') and k.startswith(key_prefix):
            state_dict[k.replace(key_prefix, '')] = xp2torch(v, device=self.device)
    if state_dict:
        backup = {k: v.clone() for k, v in self._model.state_dict().items()}
        self._model.load_state_dict(state_dict)
        yield
        self._model.load_state_dict(backup)
    else:
        yield