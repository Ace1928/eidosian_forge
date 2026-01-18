import StringIO
import time
def timervalue(self, timer='total', now=None):
    """Return the value seen by this timer so far.

    If the timer is stopped, this will be the accumulated time it has seen.
    If the timer is running, this will be the time it has seen up to now.
    If the timer has never been started, this will be zero.

    Args:
      timer: str; the name of the timer to report on.
      now: long; if provided, the time to use for 'now' for running timers.
    """
    if not now:
        now = time.time()
    if timer in self.timers:
        return self.accum.get(timer, 0.0) + (now - self.timers[timer])
    elif timer in self.accum:
        return self.accum[timer]
    else:
        return 0.0