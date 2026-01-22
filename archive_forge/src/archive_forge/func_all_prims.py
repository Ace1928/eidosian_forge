import functools
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Sequence
import torch
import torch._decomp
import torch._prims
import torch._refs
import torch._refs.nn
import torch._refs.nn.functional
import torch._refs.special
import torch.overrides
from torch._prims_common import torch_function_passthrough
@functools.lru_cache(None)
def all_prims():
    """
    Set of all prim functions, e.g., torch._prims.add in all_prims()
    """
    return {torch._prims.__dict__.get(s) for s in torch._prims.__all__}