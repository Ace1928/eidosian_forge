import collections
import warnings
from typing import Any, Iterable, Optional
from sphinx.deprecation import RemovedInSphinx70Warning
class peek_iter:
    """An iterator object that supports peeking ahead.

    Parameters
    ----------
    o : iterable or callable
        `o` is interpreted very differently depending on the presence of
        `sentinel`.

        If `sentinel` is not given, then `o` must be a collection object
        which supports either the iteration protocol or the sequence protocol.

        If `sentinel` is given, then `o` must be a callable object.

    sentinel : any value, optional
        If given, the iterator will call `o` with no arguments for each
        call to its `next` method; if the value returned is equal to
        `sentinel`, :exc:`StopIteration` will be raised, otherwise the
        value will be returned.

    See Also
    --------
    `peek_iter` can operate as a drop in replacement for the built-in
    `iter <https://docs.python.org/3/library/functions.html#iter>`_ function.

    Attributes
    ----------
    sentinel
        The value used to indicate the iterator is exhausted. If `sentinel`
        was not given when the `peek_iter` was instantiated, then it will
        be set to a new object instance: ``object()``.

    """

    def __init__(self, *args: Any) -> None:
        """__init__(o, sentinel=None)"""
        self._iterable: Iterable = iter(*args)
        self._cache: collections.deque = collections.deque()
        if len(args) == 2:
            self.sentinel = args[1]
        else:
            self.sentinel = object()

    def __iter__(self) -> 'peek_iter':
        return self

    def __next__(self, n: Optional[int]=None) -> Any:
        return self.next(n)

    def _fillcache(self, n: Optional[int]) -> None:
        """Cache `n` items. If `n` is 0 or None, then 1 item is cached."""
        if not n:
            n = 1
        try:
            while len(self._cache) < n:
                self._cache.append(next(self._iterable))
        except StopIteration:
            while len(self._cache) < n:
                self._cache.append(self.sentinel)

    def has_next(self) -> bool:
        """Determine if iterator is exhausted.

        Returns
        -------
        bool
            True if iterator has more items, False otherwise.

        Note
        ----
        Will never raise :exc:`StopIteration`.

        """
        return self.peek() != self.sentinel

    def next(self, n: Optional[int]=None) -> Any:
        """Get the next item or `n` items of the iterator.

        Parameters
        ----------
        n : int or None
            The number of items to retrieve. Defaults to None.

        Returns
        -------
        item or list of items
            The next item or `n` items of the iterator. If `n` is None, the
            item itself is returned. If `n` is an int, the items will be
            returned in a list. If `n` is 0, an empty list is returned.

        Raises
        ------
        StopIteration
            Raised if the iterator is exhausted, even if `n` is 0.

        """
        self._fillcache(n)
        if not n:
            if self._cache[0] == self.sentinel:
                raise StopIteration
            if n is None:
                result = self._cache.popleft()
            else:
                result = []
        else:
            if self._cache[n - 1] == self.sentinel:
                raise StopIteration
            result = [self._cache.popleft() for i in range(n)]
        return result

    def peek(self, n: Optional[int]=None) -> Any:
        """Preview the next item or `n` items of the iterator.

        The iterator is not advanced when peek is called.

        Returns
        -------
        item or list of items
            The next item or `n` items of the iterator. If `n` is None, the
            item itself is returned. If `n` is an int, the items will be
            returned in a list. If `n` is 0, an empty list is returned.

            If the iterator is exhausted, `peek_iter.sentinel` is returned,
            or placed as the last item in the returned list.

        Note
        ----
        Will never raise :exc:`StopIteration`.

        """
        self._fillcache(n)
        if n is None:
            result = self._cache[0]
        else:
            result = [self._cache[i] for i in range(n)]
        return result