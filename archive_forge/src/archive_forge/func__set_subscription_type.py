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
def _set_subscription_type(self, subscription_type: SubscriptionType):
    if self._subscription_type == SubscriptionType.NONE or self._subscription_type == subscription_type:
        self._subscription_type = subscription_type
    else:
        raise IllegalStateError('Subscription to topics, partitions and pattern are mutually exclusive')