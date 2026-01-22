from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch.autograd.forward_ad import _set_fwd_grad_enabled
class VmapInterpreter(FuncTorchInterpreter):

    def __init__(self, cdata: CInterpreter):
        assert cdata.key() == TransformType.Vmap
        self._cdata = cdata
        self._cptr = CVmapInterpreterPtr(cdata)

    def process(self, op, args, kwargs):
        kernel = op.functorch_table[TransformType.Vmap]
        return kernel(self, *args, **kwargs)

    def batch_size(self):
        return self._cptr.batchSize()

    def randomness(self):
        typ = self._cptr.randomness()
        if typ == RandomnessType.Error:
            return 'error'
        elif typ == RandomnessType.Same:
            return 'same'
        elif typ == RandomnessType.Different:
            return 'different'
        raise RuntimeError(f'Unknown RandomnessType: {typ}')