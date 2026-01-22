import copyreg
import enum
import functools
import warnings
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._namedtensor_internals import (
from torch.overrides import (
from torch.utils.dlpack import DLDeviceType
def _handle_torch_function_and_wrap_type_error_to_not_implemented(f):
    assigned = functools.WRAPPER_ASSIGNMENTS

    @functools.wraps(f, assigned=assigned)
    def wrapped(*args, **kwargs):
        try:
            if has_torch_function(args):
                return handle_torch_function(wrapped, args, *args, **kwargs)
            return f(*args, **kwargs)
        except TypeError:
            return NotImplemented
    return wrapped