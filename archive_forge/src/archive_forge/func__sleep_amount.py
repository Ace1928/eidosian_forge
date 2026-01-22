import threading
import time
from botocore.exceptions import CapacityNotAvailableError
def _sleep_amount(self, amount):
    return (amount - self._current_capacity) / self._fill_rate