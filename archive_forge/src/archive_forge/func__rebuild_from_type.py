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
def _rebuild_from_type(func, type, args, dict):
    if type is Tensor:
        return func(*args)
    ret = func(*args).as_subclass(type)
    ret.__dict__ = dict
    return ret