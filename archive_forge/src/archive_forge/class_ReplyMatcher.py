from contextlib import contextmanager
from itertools import count
from jeepney import HeaderFields, Message, MessageFlag, MessageType
class ReplyMatcher:

    def __init__(self):
        self._futures = {}

    @contextmanager
    def catch(self, serial, future):
        """Context manager to capture a reply for the given serial number"""
        self._futures[serial] = future
        try:
            yield future
        finally:
            del self._futures[serial]

    def dispatch(self, msg):
        """Dispatch an incoming message which may be a reply

        Returns True if a task was waiting for it, otherwise False.
        """
        rep_serial = msg.header.fields.get(HeaderFields.reply_serial, -1)
        if rep_serial in self._futures:
            self._futures[rep_serial].set_result(msg)
            return True
        else:
            return False

    def drop_all(self, exc: Exception=None):
        """Throw an error in any task still waiting for a reply"""
        if exc is None:
            exc = RouterClosed('D-Bus router closed before reply arrived')
        futures, self._futures = (self._futures, {})
        for fut in futures.values():
            fut.set_exception(exc)