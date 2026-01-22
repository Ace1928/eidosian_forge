from __future__ import annotations
import hashlib
import os
from textwrap import dedent
from typing import IO, TYPE_CHECKING
from pip._vendor.cachecontrol.cache import BaseCache, SeparateBodyBaseCache
from pip._vendor.cachecontrol.controller import CacheController
class _FileCacheMixin:
    """Shared implementation for both FileCache variants."""

    def __init__(self, directory: str, forever: bool=False, filemode: int=384, dirmode: int=448, lock_class: type[BaseFileLock] | None=None) -> None:
        try:
            if lock_class is None:
                from filelock import FileLock
                lock_class = FileLock
        except ImportError:
            notice = dedent('\n            NOTE: In order to use the FileCache you must have\n            filelock installed. You can install it via pip:\n              pip install filelock\n            ')
            raise ImportError(notice)
        self.directory = directory
        self.forever = forever
        self.filemode = filemode
        self.dirmode = dirmode
        self.lock_class = lock_class

    @staticmethod
    def encode(x: str) -> str:
        return hashlib.sha224(x.encode()).hexdigest()

    def _fn(self, name: str) -> str:
        hashed = self.encode(name)
        parts = list(hashed[:5]) + [hashed]
        return os.path.join(self.directory, *parts)

    def get(self, key: str) -> bytes | None:
        name = self._fn(key)
        try:
            with open(name, 'rb') as fh:
                return fh.read()
        except FileNotFoundError:
            return None

    def set(self, key: str, value: bytes, expires: int | datetime | None=None) -> None:
        name = self._fn(key)
        self._write(name, value)

    def _write(self, path: str, data: bytes) -> None:
        """
        Safely write the data to the given path.
        """
        try:
            os.makedirs(os.path.dirname(path), self.dirmode)
        except OSError:
            pass
        with self.lock_class(path + '.lock'):
            with _secure_open_write(path, self.filemode) as fh:
                fh.write(data)

    def _delete(self, key: str, suffix: str) -> None:
        name = self._fn(key) + suffix
        if not self.forever:
            try:
                os.remove(name)
            except FileNotFoundError:
                pass