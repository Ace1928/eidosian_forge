import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def _check_out_dims_is_int_or_int_pytree(out_dims: out_dims_t, func: Callable) -> None:
    if isinstance(out_dims, int):
        return
    tree_map_(partial(_check_int_or_none, func=func, out_dims=out_dims), out_dims)