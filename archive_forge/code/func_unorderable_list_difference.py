from collections import namedtuple, Counter
from os.path import commonprefix
def unorderable_list_difference(expected, actual):
    """Same behavior as sorted_list_difference but
    for lists of unorderable items (like dicts).

    As it does a linear search per item (remove) it
    has O(n*n) performance."""
    missing = []
    while expected:
        item = expected.pop()
        try:
            actual.remove(item)
        except ValueError:
            missing.append(item)
    return (missing, actual)