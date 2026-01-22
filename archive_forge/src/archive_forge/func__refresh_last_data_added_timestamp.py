import contextlib
from datetime import datetime
import sys
import time
def _refresh_last_data_added_timestamp(self):
    self._last_data_added_timestamp = time.time()