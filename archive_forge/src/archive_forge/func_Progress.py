from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import time
from gslib.thread_message import ProgressMessage
from gslib.utils import parallelism_framework_util
def Progress(self, bytes_processed):
    """Tracks byte processing progress, making a callback if necessary."""
    self._bytes_processed_since_callback += bytes_processed
    cur_time = time.time()
    if self._bytes_processed_since_callback > self._bytes_per_callback or (self._total_size is not None and self._total_bytes_processed + self._bytes_processed_since_callback >= self._total_size) or self._last_time - cur_time > self._timeout:
        self._total_bytes_processed += self._bytes_processed_since_callback
        if self._total_size is not None:
            bytes_sent = min(self._total_bytes_processed, self._total_size)
        else:
            bytes_sent = self._total_bytes_processed
        self._callback_func(bytes_sent, self._total_size)
        self._bytes_processed_since_callback = 0
        self._callbacks_made += 1
        self._last_time = cur_time