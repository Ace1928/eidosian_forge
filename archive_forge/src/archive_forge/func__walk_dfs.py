import contextlib
import copy
import functools
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import (
import srsly
from .backends import CupyOps, NumpyOps, Ops, ParamServer, get_current_ops
from .optimizers import Optimizer  # noqa: F401
from .shims import Shim
from .types import FloatsXd
from .util import (
def _walk_dfs(self, post_order: bool=False) -> Iterable['Model']:
    """Iterate out layers of the model, depth-first."""
    seen: Dict[int, Iterator['Model']] = dict()
    stack = [self]
    seen[id(self)] = iter(self.layers)
    if not post_order:
        yield self
    while stack:
        try:
            next_child = next(seen[id(stack[-1])])
            if not id(next_child) in seen:
                if not post_order:
                    yield next_child
                stack.append(next_child)
                seen[id(next_child)] = iter(next_child.layers)
        except StopIteration:
            if post_order:
                yield stack[-1]
            stack.pop()