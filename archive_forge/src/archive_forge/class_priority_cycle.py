from __future__ import annotations
from itertools import count
from .imports import symbol_by_name
class priority_cycle(round_robin_cycle):
    """Cycle that repeats items in order."""

    def rotate(self, last_used):
        """Unused in this implementation."""