import itertools
from .compat import collections_abc
class _SequenceIterableView(object):
    """Wrap an iterable returned by find_matches().

    This is essentially just a proxy to the underlying sequence that provides
    the same interface as `_FactoryIterableView`.
    """

    def __init__(self, sequence):
        self._sequence = sequence

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self._sequence)

    def __bool__(self):
        return bool(self._sequence)
    __nonzero__ = __bool__

    def __iter__(self):
        return iter(self._sequence)