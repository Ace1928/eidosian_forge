import itertools
from .compat import collections_abc
class _FactoryIterableView(object):
    """Wrap an iterator factory returned by `find_matches()`.

    Calling `iter()` on this class would invoke the underlying iterator
    factory, making it a "collection with ordering" that can be iterated
    through multiple times, but lacks random access methods presented in
    built-in Python sequence types.
    """

    def __init__(self, factory):
        self._factory = factory
        self._iterable = None

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, list(self))

    def __bool__(self):
        try:
            next(iter(self))
        except StopIteration:
            return False
        return True
    __nonzero__ = __bool__

    def __iter__(self):
        iterable = self._factory() if self._iterable is None else self._iterable
        self._iterable, current = itertools.tee(iterable)
        return current