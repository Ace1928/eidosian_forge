import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
class DecorateInfo:
    """Describes which test, or type of tests, should be wrapped in the given
    decorators when testing an operator. Any test that matches all provided
    arguments will be decorated. The decorators will only be applied if the
    active_if argument is True."""
    __slots__ = ['decorators', 'cls_name', 'test_name', 'device_type', 'dtypes', 'active_if']

    def __init__(self, decorators, cls_name=None, test_name=None, *, device_type=None, dtypes=None, active_if=True):
        self.decorators = list(decorators) if isinstance(decorators, collections.abc.Sequence) else [decorators]
        self.cls_name = cls_name
        self.test_name = test_name
        self.device_type = device_type
        self.dtypes = dtypes
        self.active_if = active_if
        if self.dtypes is not None:
            for dtype in self.dtypes:
                assert isinstance(dtype, torch.dtype)

    def is_active(self, cls_name, test_name, device_type, dtype, param_kwargs):
        return self.active_if and (self.cls_name is None or self.cls_name == cls_name) and (self.test_name is None or self.test_name == test_name) and (self.device_type is None or self.device_type == device_type) and (self.dtypes is None or dtype in self.dtypes) and (self.active_if(param_kwargs) if isinstance(self.active_if, Callable) else self.active_if)