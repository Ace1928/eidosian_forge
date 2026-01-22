import collections
import warnings
from typing import Any, Iterable, Optional
from sphinx.deprecation import RemovedInSphinx70Warning
def _fillcache(self, n: Optional[int]) -> None:
    """Cache `n` modified items. If `n` is 0 or None, 1 item is cached.

        Each item returned by the iterator is passed through the
        `modify_iter.modified` function before being cached.

        """
    if not n:
        n = 1
    try:
        while len(self._cache) < n:
            self._cache.append(self.modifier(next(self._iterable)))
    except StopIteration:
        while len(self._cache) < n:
            self._cache.append(self.sentinel)