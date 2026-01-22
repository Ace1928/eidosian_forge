import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
from aiokafka.coordinator.assignors.abstract import AbstractPartitionAssignor
from aiokafka.coordinator.assignors.sticky.partition_movements import PartitionMovements
from aiokafka.coordinator.assignors.sticky.sorted_set import SortedSet
from aiokafka.coordinator.protocol import (
from aiokafka.coordinator.protocol import Schema
from aiokafka.protocol.struct import Struct
from aiokafka.protocol.types import String, Array, Int32
from aiokafka.structs import TopicPartition
def _reassign_partition(self, partition):
    new_consumer = None
    for another_consumer, _ in self.sorted_current_subscriptions:
        if partition in self.consumer_to_all_potential_partitions[another_consumer]:
            new_consumer = another_consumer
            break
    assert new_consumer is not None
    self._reassign_partition_to_consumer(partition, new_consumer)