import time
import threading
def _consume_through_leaky_bucket(self):
    while not self._transfer_coordinator.exception:
        try:
            self._leaky_bucket.consume(self._bytes_seen, self._request_token)
            self._bytes_seen = 0
            return
        except RequestExceededException as e:
            self._time_utils.sleep(e.retry_time)
    else:
        raise self._transfer_coordinator.exception