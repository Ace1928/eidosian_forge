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
def _assigned_state(self, tp: TopicPartition) -> 'TopicPartitionState':
    assert self._subscription is not None
    assert self._subscription.assignment is not None
    tp_state = self._subscription.assignment.state_value(tp)
    if tp_state is None:
        raise IllegalStateError(f'No current assignment for partition {tp}')
    return tp_state