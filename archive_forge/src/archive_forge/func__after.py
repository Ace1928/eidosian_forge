import random
from collections import deque
from datetime import timedelta
from .timer import Timer
def _after(self, data):
    if data is not None:
        self._stats.append(data)