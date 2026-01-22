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
class BackOffLoopingCall(LoopingCallBase):
    """Run a method in a loop with backoff on error.

    The passed in function should return True (no error, return to
    initial_interval),
    False (error, start backing off), or raise LoopingCallDone(retvalue=None)
    (quit looping, return retvalue if set).

    When there is an error, the call will backoff on each failure. The
    backoff will be equal to double the previous base interval times some
    jitter. If a backoff would put it over the timeout, it halts immediately,
    so the call will never take more than timeout, but may and likely will
    take less time.

    When the function return value is True or False, the interval will be
    multiplied by a random jitter. If min_jitter or max_jitter is None,
    there will be no jitter (jitter=1). If min_jitter is below 0.5, the code
    may not backoff and may increase its retry rate.

    If func constantly returns True, this function will not return.

    To run a func and wait for a call to finish (by raising a LoopingCallDone):

        timer = BackOffLoopingCall(func)
        response = timer.start().wait()

    :param initial_delay: delay before first running of function
    :param starting_interval: initial interval in seconds between calls to
                              function. When an error occurs and then a
                              success, the interval is returned to
                              starting_interval
    :param timeout: time in seconds before a LoopingCallTimeout is raised.
                    The call will never take longer than timeout, but may quit
                    before timeout.
    :param max_interval: The maximum interval between calls during errors
    :param jitter: Used to vary when calls are actually run to avoid group of
                   calls all coming at the exact same time. Uses
                   random.gauss(jitter, 0.1), with jitter as the mean for the
                   distribution. If set below .5, it can cause the calls to
                   come more rapidly after each failure.
    :param min_interval: The minimum interval in seconds between calls to
                         function.
    :raises: LoopingCallTimeout if time spent doing error retries would exceed
             timeout.
    """
    _RNG = random.SystemRandom()
    _KIND = _('Dynamic backoff interval looping call')
    _RUN_ONLY_ONE_MESSAGE = _('A dynamic backoff interval looping call can only run one function at a time')

    def __init__(self, f=None, *args, **kw):
        super(BackOffLoopingCall, self).__init__(*args, f=f, **kw)
        self._error_time = 0
        self._interval = 1

    def start(self, initial_delay=None, starting_interval=1, timeout=300, max_interval=300, jitter=0.75, min_interval=0.001):
        if self._thread is not None:
            raise RuntimeError(self._RUN_ONLY_ONE_MESSAGE)
        self._error_time = 0
        self._interval = starting_interval

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
        return self._start(_idle_for, initial_delay=initial_delay)