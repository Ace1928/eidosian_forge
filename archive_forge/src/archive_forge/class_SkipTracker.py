from contextlib import contextmanager
import threading
from typing import Dict, Generator, List, Optional, Tuple
from torch import Tensor
from ..checkpoint import is_checkpointing
from ..dependency import fork, join
from ..microbatch import Batch
from ..stream import AbstractStream
from .layout import SkipLayout
from .namespace import Namespace
from .portal import Portal
class SkipTracker:
    """Tracks saved skip tensors.

    It will update the given micro-batch in place. This is because when it
    manipulates the underlying skip tensors, the current micro-batch also has
    to be connected with the skip tensors.

    One thread has one skip tracker. Call :func:`current_skip_tracker` to get
    the skip tracker on the current thread.

    """

    def __init__(self) -> None:
        self.tensors: Dict[Tuple[Namespace, str], Optional[Tensor]] = {}

    def save(self, batch: Batch, ns: Namespace, name: str, tensor: Optional[Tensor]) -> None:
        self.tensors[ns, name] = tensor

    def load(self, batch: Batch, ns: Namespace, name: str) -> Optional[Tensor]:
        return self.tensors.pop((ns, name))

    def copy(self, batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream, ns: Namespace, name: str) -> None:
        raise TypeError('copy is not supported for non-portal skip tensors')