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
def _jit_unwrap_structured(obj):
    if hasattr(obj, '_jit_unwrap'):
        return obj._jit_unwrap()
    return obj