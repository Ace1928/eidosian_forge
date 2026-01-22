from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
class EventDispatcher(BaseThread):
    """
    Consumer thread base class subclassed by event observer threads
    that dispatch events from an event queue to appropriate event handlers.

    :param timeout:
        Event queue blocking timeout (in seconds).
    :type timeout:
        ``float``
    """

    def __init__(self, timeout=DEFAULT_OBSERVER_TIMEOUT):
        BaseThread.__init__(self)
        self._event_queue = EventQueue()
        self._timeout = timeout

    @property
    def timeout(self):
        """Event queue block timeout."""
        return self._timeout

    @property
    def event_queue(self):
        """The event queue which is populated with file system events
        by emitters and from which events are dispatched by a dispatcher
        thread."""
        return self._event_queue

    def dispatch_events(self, event_queue, timeout):
        """Override this method to consume events from an event queue, blocking
        on the queue for the specified timeout before raising :class:`queue.Empty`.

        :param event_queue:
            Event queue to populate with one set of events.
        :type event_queue:
            :class:`EventQueue`
        :param timeout:
            Interval period (in seconds) to wait before timing out on the
            event queue.
        :type timeout:
            ``float``
        :raises:
            :class:`queue.Empty`
        """

    def run(self):
        while self.should_keep_running():
            try:
                self.dispatch_events(self.event_queue, self.timeout)
            except queue.Empty:
                continue