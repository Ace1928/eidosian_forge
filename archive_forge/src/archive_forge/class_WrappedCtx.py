import torch
from torch._ops import HigherOrderOperator
from torch._C._functorch import TransformType
from torch._functorch.utils import enable_single_level_autograd_function
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch._functorch.vmap import (
from torch._functorch.apis import vmap
from torch._functorch.vmap import _broadcast_to_and_flatten
from torch.autograd.forward_ad import _set_fwd_grad_enabled
from typing import Any, NamedTuple, Tuple
class WrappedCtx:
    _pt_reserved_attrs: Tuple[str, ...] = ('_pt_reserved_attrs', '_pt_inner_ctx')

    def __init__(self, ctx):
        if not isinstance(ctx, WrappedCtx):
            reserved_attrs = type(self)._pt_reserved_attrs
            for name in reserved_attrs:
                if not hasattr(ctx, name):
                    continue
                raise RuntimeError(f'PyTorch reserves the {reserved_attrs} field on ctx. Please name your fields on ctx something else to avoid name collision.')
        self._pt_inner_ctx = ctx

    def __getattr__(self, name):
        return getattr(self._pt_inner_ctx, name)

    def __setattr__(self, name, value):
        if name in type(self)._pt_reserved_attrs:
            self.__dict__[name] = value
            return
        return setattr(self._pt_inner_ctx, name, value)