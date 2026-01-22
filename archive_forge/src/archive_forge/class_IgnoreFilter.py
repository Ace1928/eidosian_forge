import os.path
import re
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Union
from .config import Config, get_xdg_config_home_path
class IgnoreFilter:

    def __init__(self, patterns: Iterable[bytes], ignorecase: bool=False, path=None) -> None:
        self._patterns: List[Pattern] = []
        self._ignorecase = ignorecase
        self._path = path
        for pattern in patterns:
            self.append_pattern(pattern)

    def append_pattern(self, pattern: bytes) -> None:
        """Add a pattern to the set."""
        self._patterns.append(Pattern(pattern, self._ignorecase))

    def find_matching(self, path: Union[bytes, str]) -> Iterable[Pattern]:
        """Yield all matching patterns for path.

        Args:
          path: Path to match
        Returns:
          Iterator over iterators
        """
        if not isinstance(path, bytes):
            path = os.fsencode(path)
        for pattern in self._patterns:
            if pattern.match(path):
                yield pattern

    def is_ignored(self, path: bytes) -> Optional[bool]:
        """Check whether a path is ignored.

        For directories, include a trailing slash.

        Returns: status is None if file is not mentioned, True if it is
            included, False if it is explicitly excluded.
        """
        status = None
        for pattern in self.find_matching(path):
            status = pattern.is_exclude
        return status

    @classmethod
    def from_path(cls, path, ignorecase: bool=False) -> 'IgnoreFilter':
        with open(path, 'rb') as f:
            return cls(read_ignore_patterns(f), ignorecase, path=path)

    def __repr__(self) -> str:
        path = getattr(self, '_path', None)
        if path is not None:
            return f'{type(self).__name__}.from_path({path!r})'
        else:
            return '<%s>' % type(self).__name__