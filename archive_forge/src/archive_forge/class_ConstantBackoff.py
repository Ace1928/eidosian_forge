import random
from abc import ABC, abstractmethod
class ConstantBackoff(AbstractBackoff):
    """Constant backoff upon failure"""

    def __init__(self, backoff):
        """`backoff`: backoff time in seconds"""
        self._backoff = backoff

    def compute(self, failures):
        return self._backoff