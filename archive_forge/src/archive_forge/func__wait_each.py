from eventlet.event import Event
from eventlet import greenthread
import collections
def _wait_each(self, pending):
    """
        When _wait_each() encounters a value of PropagateError, it raises it.

        In all other respects, _wait_each() behaves like _wait_each_raw().
        """
    for key, value in self._wait_each_raw(pending):
        yield (key, self._value_or_raise(value))