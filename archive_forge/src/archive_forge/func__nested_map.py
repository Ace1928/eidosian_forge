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
def _nested_map(condition, fn, condition_msg=None):

    def _map(obj):
        if condition(obj):
            return fn(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (list, tuple)):
            mapped = (_map(x) for x in obj)
            if hasattr(obj, '_fields'):
                return type(obj)(*mapped)
            return type(obj)(mapped)
        elif isinstance(obj, dict):
            return {x: _map(obj[x]) for x in obj}
        else:
            raise ValueError("Auto nesting doesn't know how to process an input object of type " + torch.typename(obj) + ('. Accepted types: ' + condition_msg + ', or lists/tuples of them' if condition_msg else ''))
    return _map