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
def _init_current_assignments(self, members):
    sorted_partition_consumers_by_generation = {}
    for consumer, member_metadata in members.items():
        for partitions in member_metadata.partitions:
            if partitions in sorted_partition_consumers_by_generation:
                consumers = sorted_partition_consumers_by_generation[partitions]
                if member_metadata.generation and member_metadata.generation in consumers:
                    log.warning('Partition {} is assigned to multiple consumers following sticky assignment generation {}.'.format(partitions, member_metadata.generation))
                else:
                    consumers[member_metadata.generation] = consumer
            else:
                sorted_consumers = {member_metadata.generation: consumer}
                sorted_partition_consumers_by_generation[partitions] = sorted_consumers
    for partitions, consumers in sorted_partition_consumers_by_generation.items():
        generations = sorted(consumers.keys(), reverse=True)
        self.current_assignment[consumers[generations[0]]].append(partitions)
        if len(generations) > 1:
            self.previous_assignment[partitions] = ConsumerGenerationPair(consumer=consumers[generations[1]], generation=generations[1])
    self.is_fresh_assignment = len(self.current_assignment) == 0
    for consumer_id, partitions in self.current_assignment.items():
        for partition in partitions:
            self.current_partition_consumer[partition] = consumer_id