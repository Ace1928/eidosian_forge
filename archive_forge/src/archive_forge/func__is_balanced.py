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
def _is_balanced(self):
    """Determines if the current assignment is a balanced one"""
    if len(self.current_assignment[self._get_consumer_with_least_subscriptions()]) >= len(self.current_assignment[self._get_consumer_with_most_subscriptions()]) - 1:
        return True
    all_assigned_partitions = {}
    for consumer_id, consumer_partitions in self.current_assignment.items():
        for partition in consumer_partitions:
            if partition in all_assigned_partitions:
                log.error('{} is assigned to more than one consumer.'.format(partition))
            all_assigned_partitions[partition] = consumer_id
    for consumer, _ in self.sorted_current_subscriptions:
        consumer_partition_count = len(self.current_assignment[consumer])
        if consumer_partition_count == len(self.consumer_to_all_potential_partitions[consumer]):
            continue
        for partition in self.consumer_to_all_potential_partitions[consumer]:
            if partition not in self.current_assignment[consumer]:
                other_consumer = all_assigned_partitions[partition]
                other_consumer_partition_count = len(self.current_assignment[other_consumer])
                if consumer_partition_count < other_consumer_partition_count:
                    return False
    return True