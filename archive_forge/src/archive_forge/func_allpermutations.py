import itertools
import collections
def allpermutations(orgset, k):
    """
    returns all permutations of orgset with up to k items

    :param orgset: the list to be iterated
    :param k: the maxcardinality of the subsets

    :return: an iterator of the subsets

    example:

    >>> c = allpermutations([1,2,3,4],2)
    >>> for s in c:
    ...     print(s)
    (1,)
    (2,)
    (3,)
    (4,)
    (1, 2)
    (1, 3)
    (1, 4)
    (2, 1)
    (2, 3)
    (2, 4)
    (3, 1)
    (3, 2)
    (3, 4)
    (4, 1)
    (4, 2)
    (4, 3)
    """
    return itertools.chain(*[permutation(orgset, i) for i in range(1, k + 1)])