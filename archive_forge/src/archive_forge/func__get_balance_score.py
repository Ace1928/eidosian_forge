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
@staticmethod
def _get_balance_score(assignment):
    """Calculates a balance score of a give assignment
        as the sum of assigned partitions size difference of all consumer pairs.
        A perfectly balanced assignment (with all consumers getting the same number of
        partitions) has a balance score of 0. Lower balance score indicates a more
        balanced assignment.

        Arguments:
          assignment (dict): {consumer: list of assigned topic partitions}

        Returns:
          the balance score of the assignment
        """
    score = 0
    consumer_to_assignment = {}
    for consumer_id, partitions in assignment.items():
        consumer_to_assignment[consumer_id] = len(partitions)
    consumers_to_explore = set(consumer_to_assignment.keys())
    for consumer_id in consumer_to_assignment.keys():
        if consumer_id in consumers_to_explore:
            consumers_to_explore.remove(consumer_id)
            for other_consumer_id in consumers_to_explore:
                score += abs(consumer_to_assignment[consumer_id] - consumer_to_assignment[other_consumer_id])
    return score