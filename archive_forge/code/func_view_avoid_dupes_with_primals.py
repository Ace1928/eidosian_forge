import collections
from functools import wraps
from typing import Callable, DefaultDict, Dict, List
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._logging import getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._subclasses.meta_utils import safe_is_leaf
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .functional_utils import (
from .schemas import (
from .subclass_utils import create_subclass_meta
from .utils import _get_autocast_states, KNOWN_TYPES, strict_zip
from a multi-output view call"
def view_avoid_dupes_with_primals(t):
    if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
        return transform_subclass(t, lambda _, inner_t: view_avoid_dupes_with_primals(inner_t))
    if isinstance(t, Tensor):
        return t.view(t.shape)
    return t