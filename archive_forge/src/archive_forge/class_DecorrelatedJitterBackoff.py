import random
from abc import ABC, abstractmethod
class DecorrelatedJitterBackoff(AbstractBackoff):
    """Decorrelated jitter backoff upon failure"""

    def __init__(self, cap, base):
        """
        `cap`: maximum backoff time in seconds
        `base`: base backoff time in seconds
        """
        self._cap = cap
        self._base = base
        self._previous_backoff = 0

    def reset(self):
        self._previous_backoff = 0

    def compute(self, failures):
        max_backoff = max(self._base, self._previous_backoff * 3)
        temp = random.uniform(self._base, max_backoff)
        self._previous_backoff = min(self._cap, temp)
        return self._previous_backoff