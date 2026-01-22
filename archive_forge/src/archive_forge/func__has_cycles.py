import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
def _has_cycles(self, consumer_pairs):
    cycles = set()
    for pair in consumer_pairs:
        reduced_pairs = deepcopy(consumer_pairs)
        reduced_pairs.remove(pair)
        path = [pair.src_member_id]
        if self._is_linked(pair.dst_member_id, pair.src_member_id, reduced_pairs, path) and (not self._is_subcycle(path, cycles)):
            cycles.add(tuple(path))
            log.error('A cycle of length {} was found: {}'.format(len(path) - 1, path))
    for cycle in cycles:
        if len(cycle) == 3:
            return True
    return False