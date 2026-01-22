from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch.autograd.forward_ad import _set_fwd_grad_enabled
class FuncTorchInterpreter(ABC):

    def __init__(self, cptr: Any):
        self._cptr = cptr

    @abstractmethod
    def process(self, op, args, kwargs):
        pass

    def lower(self):
        return temporarily_pop_interpreter_stack()

    def level(self):
        return self._cptr.level()

    def key(self):
        return self._cptr.key()