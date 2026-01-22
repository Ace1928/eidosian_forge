import functools
import random
import sys
import time
from eventlet import event
from eventlet import greenthread
from oslo_log import log as logging
from oslo_utils import eventletutils
from oslo_utils import excutils
from oslo_utils import reflection
from oslo_utils import timeutils
from oslo_service._i18n import _
def _idle_for(success, _elapsed):
    random_jitter = abs(self._RNG.gauss(jitter, 0.1))
    if success:
        self._interval = starting_interval
        self._error_time = 0
        return self._interval * random_jitter
    else:
        idle = max(self._interval * 2 * random_jitter, min_interval)
        idle = min(idle, max_interval)
        self._interval = max(self._interval * 2 * jitter, min_interval)
        if timeout > 0 and self._error_time + idle > timeout:
            raise LoopingCallTimeOut(_('Looping call timed out after %.02f seconds') % (self._error_time + idle))
        self._error_time += idle
        return idle