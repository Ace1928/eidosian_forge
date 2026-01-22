import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def _adjusted_atol(atol, u, v):
    u = u[0] if isinstance(u, tuple) else u
    sum_u = u.sum()
    sum_v = 1.0 if v is None else v.sum()
    return atol * float(sum_u) * float(sum_v)