import sys
import queue
import threading
from pyglet import app
from pyglet import clock
from pyglet import event
class PlatformEventLoop:
    """ Abstract class, implementation depends on platform.

    .. versionadded:: 1.2
    """

    def __init__(self):
        self._event_queue = queue.Queue()
        self._is_running = threading.Event()

    def is_running(self):
        """Return True if the event loop is currently processing, or False
        if it is blocked or not activated.

        :rtype: bool
        """
        return self._is_running.is_set()

    def post_event(self, dispatcher, event, *args):
        """Post an event into the main application thread.

        The event is queued internally until the :py:meth:`run` method's thread
        is able to dispatch the event.  This method can be safely called
        from any thread.

        If the method is called from the :py:meth:`run` method's thread (for
        example, from within an event handler), the event may be dispatched
        within the same runloop iteration or the next one; the choice is
        nondeterministic.

        :Parameters:
            `dispatcher` : EventDispatcher
                Dispatcher to process the event.
            `event` : str
                Event name.
            `args` : sequence
                Arguments to pass to the event handlers.

        """
        self._event_queue.put((dispatcher, event, args))
        self.notify()

    def dispatch_posted_events(self):
        """Immediately dispatch all pending events.

        Normally this is called automatically by the runloop iteration.
        """
        while True:
            try:
                dispatcher, evnt, args = self._event_queue.get(False)
                dispatcher.dispatch_event(evnt, *args)
            except queue.Empty:
                break
            except ReferenceError:
                pass

    def notify(self):
        """Notify the event loop that something needs processing.

        If the event loop is blocked, it will unblock and perform an iteration
        immediately.  If the event loop is running, another iteration is
        scheduled for immediate execution afterwards.
        """
        raise NotImplementedError('abstract')

    def start(self):
        pass

    def step(self, timeout=None):
        raise NotImplementedError('abstract')

    def set_timer(self, func, interval):
        pass

    def stop(self):
        pass