from __future__ import annotations
from heapq import heappop, heappush
from typing import (
from simpy.core import BoundClass, Environment
from simpy.resources import base
class PriorityItem(NamedTuple):
    """Wrap an arbitrary *item* with an order-able *priority*.

    Pairs a *priority* with an arbitrary *item*. Comparisons of *PriorityItem*
    instances only consider the *priority* attribute, thus supporting use of
    unorderable items in a :class:`PriorityStore` instance.

    """
    priority: Any
    item: Any

    def __lt__(self, other: PriorityItem) -> bool:
        return self.priority < other.priority