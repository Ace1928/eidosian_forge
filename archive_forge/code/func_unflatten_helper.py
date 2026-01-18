import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
import torch
import torch._C as _C
import torch._functorch as _functorch
import torch.utils.hooks as hooks
from torch._C import _functions
from torch._functorch.autograd_function import custom_function_call
def unflatten_helper(input, proto):
    res: List[Optional[torch.Tensor]] = []
    if hasattr(proto, '_jit_wrap'):
        return proto._jit_wrap(input)
    if not isinstance(proto, (list, tuple)):
        return (input[0], input[1:])
    for e in proto:
        if e is None:
            res.append(e)
        else:
            res_e, input = unflatten_helper(input, e)
            res.append(res_e)
    return (type(proto)(res), input)