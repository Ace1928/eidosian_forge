import logging
from typing import Set, Callable
def _check_valid(self):
    """(Debug mode only) Check "used" and "unused" sets are disjoint."""
    if self._debug_mode:
        assert self._used_uris & self._unused_uris == set()