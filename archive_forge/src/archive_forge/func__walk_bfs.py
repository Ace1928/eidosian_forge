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
def _walk_bfs(self) -> Iterable['Model']:
    """Iterate out layers of the model, breadth-first."""
    queue = [self]
    seen: Set[int] = set()
    for node in queue:
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        queue.extend(node.layers)