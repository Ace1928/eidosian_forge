import bisect
from collections import Counter, defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from nltk.internals import raise_unorderable_types, slice_bounds
class LazySubsequence(AbstractLazySequence):
    """
    A subsequence produced by slicing a lazy sequence.  This slice
    keeps a reference to its source sequence, and generates its values
    by looking them up in the source sequence.
    """
    MIN_SIZE = 100
    '\n    The minimum size for which lazy slices should be created.  If\n    ``LazySubsequence()`` is called with a subsequence that is\n    shorter than ``MIN_SIZE``, then a tuple will be returned instead.\n    '

    def __new__(cls, source, start, stop):
        """
        Construct a new slice from a given underlying sequence.  The
        ``start`` and ``stop`` indices should be absolute indices --
        i.e., they should not be negative (for indexing from the back
        of a list) or greater than the length of ``source``.
        """
        if stop - start < cls.MIN_SIZE:
            return list(islice(source.iterate_from(start), stop - start))
        else:
            return object.__new__(cls)

    def __init__(self, source, start, stop):
        self._source = source
        self._start = start
        self._stop = stop

    def __len__(self):
        return self._stop - self._start

    def iterate_from(self, start):
        return islice(self._source.iterate_from(start + self._start), max(0, len(self) - start))