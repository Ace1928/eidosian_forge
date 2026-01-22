from __future__ import annotations
from heapq import heappop, heappush
from typing import (
from simpy.core import BoundClass, Environment
from simpy.resources import base
class StorePut(base.Put):
    """Request to put *item* into the *store*. The request is triggered once
    there is space for the item in the store.

    """

    def __init__(self, store: Store, item: Any):
        self.item = item
        'The item to put into the store.'
        super().__init__(store)