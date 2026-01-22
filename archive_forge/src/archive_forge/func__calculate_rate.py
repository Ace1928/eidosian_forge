import time
import threading
def _calculate_rate(self, amt, time_at_consumption):
    time_delta = time_at_consumption - self._last_time
    if time_delta <= 0:
        return float('inf')
    return amt / time_delta