import copy
import dataclasses
import functools
from typing import (
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from .graph_signature import (  # noqa: F401
def _disable_prexisiting_fake_mode(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with maybe_disable_fake_tensor_mode():
            return fn(*args, **kwargs)
    return wrapper