import torch
from collections import OrderedDict
import weakref
import warnings
from typing import Any, Tuple
def _set_user_hook(self, grad_fn):

    def hook(grad_input, _):
        if self.grad_outputs is None:
            return
        res = self._pack_with_none(self.input_tensors_index, grad_input, self.n_inputs)
        for hook in self.user_hooks:
            out = hook(self.module, res, self.grad_outputs)
            if out is None:
                continue
            if len(out) != len(res):
                raise RuntimeError(f'Backward hook returned an invalid number of grad_input, got {len(out)}, but expected {len(res)}')
            res = out
        self.grad_outputs = None
        return self._unpack_none(self.input_tensors_index, res)
    grad_fn.register_hook(hook)