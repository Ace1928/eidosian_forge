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
@contextmanager
def use_skip_tracker(skip_tracker: SkipTracker) -> Generator[None, None, None]:
    """Registers the given skip tracker on the current thread within a
    context::

        with use_skip_tracker(my_skip_tracker):
            ...

    """
    orig = thread_local.skip_tracker
    thread_local.skip_tracker = skip_tracker
    try:
        yield
    finally:
        thread_local.skip_tracker = orig