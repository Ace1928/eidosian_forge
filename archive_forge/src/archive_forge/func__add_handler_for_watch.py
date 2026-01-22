from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
def _add_handler_for_watch(self, event_handler, watch):
    if watch not in self._handlers:
        self._handlers[watch] = set()
    self._handlers[watch].add(event_handler)