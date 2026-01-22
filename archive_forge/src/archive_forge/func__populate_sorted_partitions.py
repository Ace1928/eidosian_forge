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
def _populate_sorted_partitions(self):
    all_partitions = set(((tp, tuple(consumers)) for tp, consumers in self.partition_to_all_potential_consumers.items()))
    partitions_sorted_by_num_of_potential_consumers = sorted(all_partitions, key=partitions_comparator_key)
    self.sorted_partitions = []
    if not self.is_fresh_assignment and self._are_subscriptions_identical():
        assignments = deepcopy(self.current_assignment)
        for partitions in assignments.values():
            to_remove = []
            for partition in partitions:
                if partition not in self.partition_to_all_potential_consumers:
                    to_remove.append(partition)
            for partition in to_remove:
                partitions.remove(partition)
        sorted_consumers = SortedSet(iterable=[(consumer, tuple(partitions)) for consumer, partitions in assignments.items()], key=subscriptions_comparator_key)
        while sorted_consumers:
            consumer, _ = sorted_consumers.pop_last()
            remaining_partitions = assignments[consumer]
            previous_partitions = set(self.previous_assignment.keys()).intersection(set(remaining_partitions))
            if previous_partitions:
                partition = previous_partitions.pop()
                remaining_partitions.remove(partition)
                self.sorted_partitions.append(partition)
                sorted_consumers.add((consumer, tuple(assignments[consumer])))
            elif remaining_partitions:
                self.sorted_partitions.append(remaining_partitions.pop())
                sorted_consumers.add((consumer, tuple(assignments[consumer])))
        while partitions_sorted_by_num_of_potential_consumers:
            partition = partitions_sorted_by_num_of_potential_consumers.pop(0)[0]
            if partition not in self.sorted_partitions:
                self.sorted_partitions.append(partition)
    else:
        while partitions_sorted_by_num_of_potential_consumers:
            self.sorted_partitions.append(partitions_sorted_by_num_of_potential_consumers.pop(0)[0])