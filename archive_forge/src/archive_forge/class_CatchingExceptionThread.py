import sys
import threading
from typing import Callable, List, Optional, Type, Union
class CatchingExceptionThread(threading.Thread):
    """A thread that keeps track of exceptions.

    If an exception occurs during the thread execution, it's caught and
    re-raised when the thread is joined().
    """
    ignored_exceptions: Optional[Callable[[Exception], bool]]

    def __init__(self, *args, **kwargs):
        try:
            sync_event = kwargs.pop('sync_event')
        except KeyError:
            sync_event = threading.Event()
        super().__init__(*args, **kwargs)
        self.set_sync_event(sync_event)
        self.exception = None
        self.ignored_exceptions = None
        self.lock = threading.Lock()

    def set_sync_event(self, event):
        """Set the ``sync_event`` event used to synchronize exception catching.

        When the thread uses an event to synchronize itself with another thread
        (setting it when the other thread can wake up from a ``wait`` call),
        the event must be set after catching an exception or the other thread
        will hang.

        Some threads require multiple events and should set the relevant one
        when appropriate.

        Note that the event should be initially cleared so the caller can
        wait() on him and be released when the thread set the event.

        Also note that the thread can use multiple events, setting them as it
        progress, while the caller can chose to wait on any of them. What
        matters is that there is always one event set so that the caller is
        always released when an exception is caught. Re-using the same event is
        therefore risky as the thread itself has no idea about which event the
        caller is waiting on. If the caller has already been released then a
        cleared event won't guarantee that the caller is still waiting on it.
        """
        self.sync_event = event

    def switch_and_set(self, new):
        """Switch to a new ``sync_event`` and set the current one.

        Using this method protects against race conditions while setting a new
        ``sync_event``.

        Note that this allows a caller to wait either on the old or the new
        event depending on whether it wants a fine control on what is happening
        inside a thread.

        :param new: The event that will become ``sync_event``
        """
        cur = self.sync_event
        self.lock.acquire()
        try:
            try:
                self.set_sync_event(new)
            except BaseException:
                self.set_sync_event(cur)
                raise
            cur.set()
        finally:
            self.lock.release()

    def set_ignored_exceptions(self, ignored: Union[Callable[[Exception], bool], None, List[Type[Exception]], Type[Exception]]):
        """Declare which exceptions will be ignored.

        :param ignored: Can be either:

           - None: all exceptions will be raised,
           - an exception class: the instances of this class will be ignored,
           - a tuple of exception classes: the instances of any class of the
             list will be ignored,
           - a callable: that will be passed the exception object
             and should return True if the exception should be ignored
        """
        if ignored is None:
            self.ignored_exceptions = None
        elif isinstance(ignored, (Exception, tuple)):
            self.ignored_exceptions = lambda e: isinstance(e, ignored)
        elif isinstance(ignored, list):
            self.ignored_exceptions = lambda e: isinstance(e, tuple(ignored))
        else:
            self.ignored_exceptions = ignored

    def run(self):
        """Overrides Thread.run to capture any exception."""
        self.sync_event.clear()
        try:
            try:
                super().run()
            except BaseException:
                self.exception = sys.exc_info()
        finally:
            self.sync_event.set()

    def join(self, timeout=None):
        """Overrides Thread.join to raise any exception caught.

        Calling join(timeout=0) will raise the caught exception or return None
        if the thread is still alive.
        """
        super().join(timeout)
        if self.exception is not None:
            exc_class, exc_value, exc_tb = self.exception
            self.exception = None
            if self.ignored_exceptions is None or not self.ignored_exceptions(exc_value):
                raise exc_value

    def pending_exception(self):
        """Raise the caught exception.

        This does nothing if no exception occurred.
        """
        self.join(timeout=0)