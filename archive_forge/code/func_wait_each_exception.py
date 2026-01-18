from eventlet.event import Event
from eventlet import greenthread
import collections
def wait_each_exception(self, keys=_MISSING):
    """
        wait_each_exception() filters results so that only exceptions are
        yielded. Not every provided (or defaulted) key will necessarily be
        represented, though naturally the generator will not finish until
        all have completed.

        Unlike other DAGPool methods, wait_each_exception() simply yields
        :class:`PropagateError` instances as values rather than raising them.

        In all other respects, wait_each_exception() behaves like :meth:`wait_each`.
        """
    for key, value in self._wait_each_raw(self._get_keyset_for_wait_each(keys)):
        if isinstance(value, PropagateError):
            yield (key, value)