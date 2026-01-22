from __future__ import annotations
from itertools import count
from .imports import symbol_by_name
class round_robin_cycle:
    """Iterator that cycles between items in round-robin."""

    def __init__(self, it=None):
        self.items = it if it is not None else []

    def update(self, it):
        """Update items from iterable."""
        self.items[:] = it

    def consume(self, n):
        """Consume n items."""
        return self.items[:n]

    def rotate(self, last_used):
        """Move most recently used item to end of list."""
        items = self.items
        try:
            items.append(items.pop(items.index(last_used)))
        except ValueError:
            pass
        return last_used