from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Type
from simpy.core import BoundClass, Environment, SimTime
from simpy.resources import base
class SortedQueue(list):
    """Queue for sorting events by their :attr:`~PriorityRequest.key`
    attribute.

    """

    def __init__(self, maxlen: Optional[int]=None):
        super().__init__()
        self.maxlen = maxlen
        'Maximum length of the queue.'

    def append(self, item: Any) -> None:
        """Sort *item* into the queue.

        Raise a :exc:`RuntimeError` if the queue is full.

        """
        if self.maxlen is not None and len(self) >= self.maxlen:
            raise RuntimeError('Cannot append event. Queue is full.')
        super().append(item)
        super().sort(key=lambda e: e.key)