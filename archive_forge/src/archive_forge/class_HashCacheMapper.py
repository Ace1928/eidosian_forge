from __future__ import annotations
import abc
import hashlib
from fsspec.implementations.local import make_path_posix
class HashCacheMapper(AbstractCacheMapper):
    """Cache mapper that uses a hash of the remote URL."""

    def __call__(self, path: str) -> str:
        return hashlib.sha256(path.encode()).hexdigest()