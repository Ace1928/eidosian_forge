import itertools
import collections
def __combination(orgset, k):
    """
    fall back if probstat is not installed note it is GPL so cannot
    be included
    """
    if k == 1:
        for i in orgset:
            yield (i,)
    elif k > 1:
        for i, x in enumerate(orgset):
            for s in __combination(orgset[i + 1:], k - 1):
                yield ((x,) + s)