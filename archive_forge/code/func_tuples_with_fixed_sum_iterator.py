import re
def tuples_with_fixed_sum_iterator(N, l, skipVertices=False):
    """
    Iterates through all l-tuples of non-negative integers summing up to N in
    lexicographic order. If skipVertices is True, N-tuples containing N, i.e.,
    of the form (0...0,N,0...0), are skipped.

    >>> list(tuples_with_fixed_sum_iterator(2, 3))
    [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    >>> list(tuples_with_fixed_sum_iterator(2, 3, skipVertices = True))
    [(0, 1, 1), (1, 0, 1), (1, 1, 0)]

    """
    for i in _lists_with_fixed_sum_iterator(N, l):
        if not (skipVertices and N in i):
            yield tuple(i)