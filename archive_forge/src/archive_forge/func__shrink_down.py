from __future__ import annotations
import os
from collections import deque
from queue import Empty
from queue import LifoQueue as _LifoQueue
from typing import TYPE_CHECKING
from . import exceptions
from .utils.compat import register_after_fork
from .utils.functional import lazy
def _shrink_down(self, collect=True):

    class Noop:

        def __enter__(self):
            pass

        def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: TracebackType) -> None:
            pass
    resource = self._resource
    with getattr(resource, 'mutex', Noop()):
        while len(resource.queue) > self.limit:
            R = resource.queue.popleft()
            if collect:
                self.collect_resource(R)