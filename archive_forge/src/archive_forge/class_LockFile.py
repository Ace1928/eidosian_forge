from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
class LockFile:
    """Provides methods to obtain, check for, and release a file based lock which
    should be used to handle concurrent access to the same file.

    As we are a utility class to be derived from, we only use protected methods.

    Locks will automatically be released on destruction.
    """
    __slots__ = ('_file_path', '_owns_lock')

    def __init__(self, file_path: PathLike) -> None:
        self._file_path = file_path
        self._owns_lock = False

    def __del__(self) -> None:
        self._release_lock()

    def _lock_file_path(self) -> str:
        """:return: Path to lockfile"""
        return '%s.lock' % self._file_path

    def _has_lock(self) -> bool:
        """
        :return: True if we have a lock and if the lockfile still exists

        :raise AssertionError: If our lock-file does not exist
        """
        return self._owns_lock

    def _obtain_lock_or_raise(self) -> None:
        """Create a lock file as flag for other instances, mark our instance as lock-holder.

        :raise IOError: If a lock was already present or a lock file could not be written
        """
        if self._has_lock():
            return
        lock_file = self._lock_file_path()
        if osp.isfile(lock_file):
            raise IOError('Lock for file %r did already exist, delete %r in case the lock is illegal' % (self._file_path, lock_file))
        try:
            with open(lock_file, mode='w'):
                pass
        except OSError as e:
            raise IOError(str(e)) from e
        self._owns_lock = True

    def _obtain_lock(self) -> None:
        """The default implementation will raise if a lock cannot be obtained.
        Subclasses may override this method to provide a different implementation."""
        return self._obtain_lock_or_raise()

    def _release_lock(self) -> None:
        """Release our lock if we have one."""
        if not self._has_lock():
            return
        lfp = self._lock_file_path()
        try:
            rmfile(lfp)
        except OSError:
            pass
        self._owns_lock = False