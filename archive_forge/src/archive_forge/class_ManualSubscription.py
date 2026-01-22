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
class ManualSubscription(Subscription):
    """ Describes a user assignment
    """

    def __init__(self, user_assignment: Iterable[TopicPartition], loop=None):
        topics = (tp.topic for tp in user_assignment)
        super().__init__(topics, loop=loop)
        self._assignment = Assignment(user_assignment)

    def _assign(self, topic_partitions: Set[TopicPartition]):
        assert False, 'Should not be called'

    @property
    def _reassignment_in_progress(self):
        return False

    @_reassignment_in_progress.setter
    def _reassignment_in_progress(self, value):
        pass

    def _begin_reassignment(self):
        assert False, 'Should not be called'