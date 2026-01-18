import abc
import contextlib
import weakref
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
def unpack_hook(tup):
    handle = tup
    error_msg = "Trying to backward outside of the 'allow_mutation_on_saved_tensors' contextin which the graph was originally recorded."
    assert _allow_mutation_on_saved_tensors_enabled, error_msg
    if handle in ctx.cloned:
        res = ctx.cloned[handle]
    else:
        assert handle in ctx.original, error_msg
        res = ctx.original[handle]
    return res