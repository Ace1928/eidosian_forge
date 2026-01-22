from __future__ import annotations
import abc
import hashlib
from fsspec.implementations.local import make_path_posix
class BasenameCacheMapper(AbstractCacheMapper):
    """Cache mapper that uses the basename of the remote URL and a fixed number
    of directory levels above this.

    The default is zero directory levels, meaning different paths with the same
    basename will have the same cached basename.
    """

    def __init__(self, directory_levels: int=0):
        if directory_levels < 0:
            raise ValueError('BasenameCacheMapper requires zero or positive directory_levels')
        self.directory_levels = directory_levels
        self._separator = '_@_'

    def __call__(self, path: str) -> str:
        path = make_path_posix(path)
        prefix, *bits = path.rsplit('/', self.directory_levels + 1)
        if bits:
            return self._separator.join(bits)
        else:
            return prefix

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and self.directory_levels == other.directory_levels

    def __hash__(self) -> int:
        return super().__hash__() ^ hash(self.directory_levels)