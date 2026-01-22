import dataclasses
import warnings
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import py_sym_types
def _get_symint_hints(exprs):
    """
    Get the hints of a list/tuple of int/SymInt.
    """
    if isinstance(exprs, (list, tuple)):
        return type(exprs)((_get_symint_hints(e) for e in exprs))
    elif isinstance(exprs, torch.SymInt):
        return exprs.node.shape_env.size_hint(exprs.node.expr)
    else:
        return exprs