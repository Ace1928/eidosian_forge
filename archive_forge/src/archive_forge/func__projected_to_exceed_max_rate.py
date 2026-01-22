import time
import threading
def _projected_to_exceed_max_rate(self, amt, time_now):
    projected_rate = self._rate_tracker.get_projected_rate(amt, time_now)
    return projected_rate > self._max_rate