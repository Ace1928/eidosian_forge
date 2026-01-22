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
def _can_consumer_participate_in_reassignment(self, consumer):
    current_partitions = self.current_assignment[consumer]
    current_assignment_size = len(current_partitions)
    max_assignment_size = len(self.consumer_to_all_potential_partitions[consumer])
    if current_assignment_size > max_assignment_size:
        log.error('The consumer {} is assigned more partitions than the maximum possible.'.format(consumer))
    if current_assignment_size < max_assignment_size:
        return True
    for partition in current_partitions:
        if self._can_partition_participate_in_reassignment(partition):
            return True
    return False