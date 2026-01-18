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
def tt(t):

    def _tt(t):
        with torch.no_grad():
            return f(t)
    if isinstance(t, torch.Tensor):
        return _tt(t)
    elif isinstance(t, torch.dtype):
        return _tt(t)
    elif isinstance(t, list):
        return list(map(tt, t))
    elif isinstance(t, tuple):
        return tuple(map(tt, t))
    elif isinstance(t, dict):
        return {k: tt(v) for k, v in t.items()}
    else:
        return t