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
def _move_partition(self, partition, new_consumer):
    old_consumer = self.current_partition_consumer[partition]
    self._remove_consumer_from_current_subscriptions_and_maintain_order(old_consumer)
    self._remove_consumer_from_current_subscriptions_and_maintain_order(new_consumer)
    self.partition_movements.move_partition(partition, old_consumer, new_consumer)
    self.current_assignment[old_consumer].remove(partition)
    self.current_assignment[new_consumer].append(partition)
    self.current_partition_consumer[partition] = new_consumer
    self._add_consumer_to_current_subscriptions_and_maintain_order(new_consumer)
    self._add_consumer_to_current_subscriptions_and_maintain_order(old_consumer)