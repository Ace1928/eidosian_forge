import time
import threading
def _raise_request_exceeded_exception(self, amt, request_token, time_now):
    allocated_time = amt / float(self._max_rate)
    retry_time = self._consumption_scheduler.schedule_consumption(amt, request_token, allocated_time)
    raise RequestExceededException(requested_amt=amt, retry_time=retry_time)