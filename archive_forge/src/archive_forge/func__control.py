import os
import sys
from eventlet import patcher, support
from eventlet.hubs import hub
def _control(self, events, max_events, timeout):
    try:
        return self.kqueue.control(events, max_events, timeout)
    except OSError:
        if os.getpid() != self._pid:
            self._reinit_kqueue()
            return self.kqueue.control(events, max_events, timeout)
        raise