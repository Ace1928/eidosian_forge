import logging
import contextlib
import copy
import time
from asyncio import shield, Event, Future
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Pattern, Set
from aiokafka.errors import IllegalStateError
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.abc import ConsumerRebalanceListener
from aiokafka.util import create_future, get_running_loop
def update_committed(self, offset_meta: OffsetAndMetadata):
    """ Called by Coordinator on successful commit to update commit cache.
        """
    for fut in self._committed_futs:
        if not fut.done():
            fut.set_result(offset_meta)
    self._committed_futs.clear()