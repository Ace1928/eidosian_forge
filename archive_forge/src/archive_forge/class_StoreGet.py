from __future__ import annotations
from heapq import heappop, heappush
from typing import (
from simpy.core import BoundClass, Environment
from simpy.resources import base
class StoreGet(base.Get):
    """Request to get an *item* from the *store*. The request is triggered
    once there is an item available in the store.

    """