import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
def _inplace_wrapper(fn):
    """
    Given a nn.functional non-linearity, implements its `inplace: bool` argument
    """

    @wraps(fn)
    def _fn(a, *args, inplace=False, **kwargs):
        if inplace:
            torch._check('out' not in kwargs, lambda: 'Cannot set inplace=True and pass out= at the same time')
            return fn(a, *args, inplace=False, out=a, **kwargs)
        else:
            return fn(a, *args, inplace=False, **kwargs)
    return _fn