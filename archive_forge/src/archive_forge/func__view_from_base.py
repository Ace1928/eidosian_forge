import contextlib
import warnings
import weakref
from typing import ContextManager, List, Optional, Tuple, TYPE_CHECKING
import torch
from torch._C._functorch import (
from torch._guards import Source
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from torch.utils.weak import WeakIdRef
import torch._prims_common as utils
def _view_from_base(base, t):
    if t.is_nested:
        return t._view_func_unsafe(base)
    else:
        sizes, strides, storage_offset = sym_sizes_strides_storage_offset(t, source)
        return base.as_strided(sizes, strides, storage_offset)