from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
def unschedule(self, watch):
    """Unschedules a watch.

        :param watch:
            The watch to unschedule.
        :type watch:
            An instance of :class:`ObservedWatch` or a subclass of
            :class:`ObservedWatch`
        """
    with self._lock:
        emitter = self._emitter_for_watch[watch]
        del self._handlers[watch]
        self._remove_emitter(emitter)
        self._watches.remove(watch)