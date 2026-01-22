from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
def _remove_emitter(self, emitter):
    del self._emitter_for_watch[emitter.watch]
    self._emitters.remove(emitter)
    emitter.stop()
    try:
        emitter.join()
    except RuntimeError:
        pass