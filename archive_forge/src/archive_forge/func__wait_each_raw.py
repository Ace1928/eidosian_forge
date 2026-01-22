from eventlet.event import Event
from eventlet import greenthread
import collections
def _wait_each_raw(self, pending):
    """
        pending is a set() of keys for which we intend to wait. THIS SET WILL
        BE DESTRUCTIVELY MODIFIED: as each key acquires a value, that key will
        be removed from the passed 'pending' set.

        _wait_each_raw() does not treat a PropagateError instance specially:
        it will be yielded to the caller like any other value.

        In all other respects, _wait_each_raw() behaves like wait_each().
        """
    while True:
        for key in pending.copy():
            value = self.values.get(key, _MISSING)
            if value is not _MISSING:
                pending.remove(key)
                yield (key, value)
        if not pending:
            break
        self.event.wait()