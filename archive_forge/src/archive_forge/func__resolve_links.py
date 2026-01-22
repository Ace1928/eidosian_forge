from collections import deque
import sys
from greenlet import GreenletExit
from eventlet import event
from eventlet import hubs
from eventlet import support
from eventlet import timeout
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
import warnings
def _resolve_links(self):
    if self._resolving_links:
        return
    if not self._exit_funcs:
        return
    self._resolving_links = True
    try:
        while self._exit_funcs:
            f, ca, ckw = self._exit_funcs.popleft()
            f(self, *ca, **ckw)
    finally:
        self._resolving_links = False