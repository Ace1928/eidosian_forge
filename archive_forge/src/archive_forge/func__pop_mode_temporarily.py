import contextlib
from typing import Optional, Union, List, Set, Dict, Any
import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
@contextlib.contextmanager
def _pop_mode_temporarily(k: Optional[DispatchKey]=None):
    old = _pop_mode(k)
    try:
        yield old
    finally:
        _push_mode(old, k)