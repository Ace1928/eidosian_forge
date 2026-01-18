from eventlet.event import Event
from eventlet import greenthread
import collections
def waiting_for(self, key=_MISSING):
    """
        waiting_for(key) returns a set() of the keys for which the DAGPool
        greenthread spawned with that *key* is still waiting. If you pass a
        *key* for which no greenthread was spawned, waiting_for() raises
        KeyError.

        waiting_for() without argument returns a dict. Its keys are the keys
        of DAGPool greenthreads still waiting on one or more values. In the
        returned dict, the value of each such key is the set of other keys for
        which that greenthread is still waiting.

        This method allows diagnosing a "hung" DAGPool. If certain
        greenthreads are making no progress, it's possible that they are
        waiting on keys for which there is no greenthread and no :meth:`post` data.
        """
    available = set(self.values.keys())
    if key is not _MISSING:
        coro = self.coros.get(key, _MISSING)
        if coro is _MISSING:
            self.values[key]
            return set()
        else:
            return coro.pending - available
    return {key: pending for key, pending in ((key, coro.pending - available) for key, coro in self.coros.items()) if pending}