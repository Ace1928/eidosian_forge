import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@staticmethod
def _find_permutations(top_partitions, bottom_partitions):
    """
        Return the pairs of top/bottom partitions where the partitions are
        different. Ensures that all partitions in both top and bottom
        partitions have size 1.
        """
    permutations = set()
    for top, bot in zip(top_partitions, bottom_partitions):
        if len(top) != 1 or len(bot) != 1:
            raise IndexError(f'Not all nodes are coupled. This is impossible: {top_partitions}, {bottom_partitions}')
        if top != bot:
            permutations.add(frozenset((next(iter(top)), next(iter(bot)))))
    return permutations