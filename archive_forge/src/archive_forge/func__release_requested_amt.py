import time
import threading
def _release_requested_amt(self, amt, time_now):
    self._rate_tracker.record_consumption_rate(amt, time_now)
    return amt