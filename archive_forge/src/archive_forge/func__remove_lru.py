from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar
def _remove_lru(self) -> None:
    """Remove one entry from the lru, and handle consequences.

        If there are no more references to the lru, then this entry should be
        removed from the cache.
        """
    assert self._least_recently_used
    self._remove_node(self._least_recently_used)