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
def warn_bc_breaking():
    warnings.warn('Backwards compatibility: New undefined gradient support checking feature is enabled by default, but it may break existing callers of this function. If this is true for you, you can call this function with "check_undefined_grad=False" to disable the feature')