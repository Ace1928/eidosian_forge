from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
def _clear_emitters(self):
    for emitter in self._emitters:
        emitter.stop()
    for emitter in self._emitters:
        try:
            emitter.join()
        except RuntimeError:
            pass
    self._emitters.clear()
    self._emitter_for_watch.clear()