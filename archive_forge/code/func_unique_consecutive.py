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
def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
    """Eliminates all but the first element from every consecutive group of equivalent elements.

        See :func:`torch.unique_consecutive`
        """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.unique_consecutive, (self,), self, return_inverse=return_inverse, return_counts=return_counts, dim=dim)
    return torch.unique_consecutive(self, return_inverse=return_inverse, return_counts=return_counts, dim=dim)