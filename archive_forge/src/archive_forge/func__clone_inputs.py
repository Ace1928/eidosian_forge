import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
def _clone_inputs(args):

    def clone_input(a):
        if a is None:
            return None
        elif isinstance(a, torch.Tensor):
            v = a.detach().clone(memory_format=None if a.is_mkldnn else torch.preserve_format).requires_grad_(a.requires_grad)
            if a.grad is not None:
                v.grad = clone_input(v.grad)
            return v
        else:
            return a.clone(memory_format=torch.preserve_format)
    return function._nested_map(lambda x: isinstance(x, torch.Tensor), clone_input, condition_msg='tensors')(args)