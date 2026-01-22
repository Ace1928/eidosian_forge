import random
from abc import ABC, abstractmethod
class FullJitterBackoff(AbstractBackoff):
    """Full jitter backoff upon failure"""

    def __init__(self, cap, base):
        """
        `cap`: maximum backoff time in seconds
        `base`: base backoff time in seconds
        """
        self._cap = cap
        self._base = base

    def compute(self, failures):
        return random.uniform(0, min(self._cap, self._base * 2 ** failures))