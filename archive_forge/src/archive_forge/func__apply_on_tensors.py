import torch
from collections import OrderedDict
import weakref
import warnings
from typing import Any, Tuple
def _apply_on_tensors(self, fn, args):
    tensors_idx = []
    tensors = []
    requires_grad = False
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            tensors_idx.append(i)
            tensors.append(arg)
            requires_grad |= arg.requires_grad
    if not (requires_grad and torch.is_grad_enabled()):
        return (args, None)
    new_tensors = torch.nn.modules._functions.BackwardHookFunction.apply(*tensors)
    if len(new_tensors) == 0:
        raise RuntimeError('Cannot set Module backward hook for a Module with no input Tensors.')
    grad_fns = [t.grad_fn for t in new_tensors if t.grad_fn is not None and t.grad_fn.name() == 'BackwardHookFunctionBackward']
    if len(grad_fns) == 0:
        raise RuntimeError('Error while setting up backward hooks. Please open an issue with a code sample to reproduce this.')
    fn(grad_fns[0])
    arg_list = list(args)
    for idx, val in zip(tensors_idx, new_tensors):
        arg_list[idx] = val
    if type(args) is tuple:
        out = tuple(arg_list)
    else:
        out = type(args)(*arg_list)
    return (out, tensors_idx)