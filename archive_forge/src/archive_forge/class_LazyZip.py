import bisect
from collections import Counter, defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from nltk.internals import raise_unorderable_types, slice_bounds
class LazyZip(LazyMap):
    """
    A lazy sequence whose elements are tuples, each containing the i-th
    element from each of the argument sequences.  The returned list is
    truncated in length to the length of the shortest argument sequence. The
    tuples are constructed lazily -- i.e., when you read a value from the
    list, ``LazyZip`` will calculate that value by forming a tuple from
    the i-th element of each of the argument sequences.

    ``LazyZip`` is essentially a lazy version of the Python primitive function
    ``zip``.  In particular, an evaluated LazyZip is equivalent to a zip:

        >>> from nltk.collections import LazyZip
        >>> sequence1, sequence2 = [1, 2, 3], ['a', 'b', 'c']
        >>> zip(sequence1, sequence2) # doctest: +SKIP
        [(1, 'a'), (2, 'b'), (3, 'c')]
        >>> list(LazyZip(sequence1, sequence2))
        [(1, 'a'), (2, 'b'), (3, 'c')]
        >>> sequences = [sequence1, sequence2, [6,7,8,9]]
        >>> list(zip(*sequences)) == list(LazyZip(*sequences))
        True

    Lazy zips can be useful for conserving memory in cases where the argument
    sequences are particularly long.

    A typical example of a use case for this class is combining long sequences
    of gold standard and predicted values in a classification or tagging task
    in order to calculate accuracy.  By constructing tuples lazily and
    avoiding the creation of an additional long sequence, memory usage can be
    significantly reduced.
    """

    def __init__(self, *lists):
        """
        :param lists: the underlying lists
        :type lists: list(list)
        """
        LazyMap.__init__(self, lambda *elts: elts, *lists)

    def iterate_from(self, index):
        iterator = LazyMap.iterate_from(self, index)
        while index < len(self):
            yield next(iterator)
            index += 1
        return

    def __len__(self):
        return min((len(lst) for lst in self._lists))