from collections import deque
from contextlib import contextmanager
import sys
import time
from eventlet.pools import Pool
from eventlet import timeout
from eventlet import hubs
from eventlet.hubs.timer import Timer
from eventlet.greenthread import GreenThread
def _schedule_expiration(self):
    """Sets up a timer that will call _expire_old_connections when the
        oldest connection currently in the free pool is ready to expire.  This
        is the earliest possible time that a connection could expire, thus, the
        timer will be running as infrequently as possible without missing a
        possible expiration.

        If this function is called when a timer is already scheduled, it does
        nothing.

        If max_age or max_idle is 0, _schedule_expiration likewise does nothing.
        """
    if self.max_age == 0 or self.max_idle == 0:
        return
    if self._expiration_timer is not None and (not getattr(self._expiration_timer, 'called', False)):
        return
    try:
        now = time.time()
        self._expire_old_connections(now)
        idle_delay = self.free_items[-1][0] - now + self.max_idle
        oldest = min([t[1] for t in self.free_items])
        age_delay = oldest - now + self.max_age
        next_delay = min(idle_delay, age_delay)
    except (IndexError, ValueError):
        self._expiration_timer = None
        return
    if next_delay > 0:
        self._expiration_timer = Timer(next_delay, GreenThread(hubs.get_hub().greenlet).switch, self._schedule_expiration, [], {})
        self._expiration_timer.schedule()