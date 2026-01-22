import collections
import contextlib
import itertools
import queue
import threading
import time
import memcache
from oslo_log import log
from oslo_cache._i18n import _
from oslo_cache import exception
def _drop_expired_connections(self):
    """Drop all expired connections from the left end of the queue."""
    now = time.time()
    try:
        while self.queue[0].ttl < now:
            conn = self.queue.popleft().connection
            self._trace_logger('Reaping connection %s', id(conn))
            self._destroy_connection(conn)
    except IndexError:
        pass