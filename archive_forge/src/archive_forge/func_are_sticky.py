import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
def are_sticky(self):
    for topic, movements in self.partition_movements_by_topic.items():
        movement_pairs = set(movements.keys())
        if self._has_cycles(movement_pairs):
            log.error('Stickiness is violated for topic {}\nPartition movements for this topic occurred among the following consumer pairs:\n{}'.format(topic, movement_pairs))
            return False
    return True