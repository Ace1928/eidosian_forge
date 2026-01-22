import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class LockableIniFileStore(TransportIniFileStore):
    """A ConfigObjStore using locks on save to ensure store integrity."""

    def __init__(self, transport, file_name, lock_dir_name=None):
        """A config Store using ConfigObj for storage.

        Args:
          transport: The transport object where the config file is located.
          file_name: The config file basename in the transport directory.
        """
        if lock_dir_name is None:
            lock_dir_name = 'lock'
        self.lock_dir_name = lock_dir_name
        super().__init__(transport, file_name)
        self._lock = lockdir.LockDir(self.transport, self.lock_dir_name)

    def lock_write(self, token=None):
        """Takes a write lock in the directory containing the config file.

        If the directory doesn't exist it is created.
        """
        self.transport.create_prefix()
        token = self._lock.lock_write(token)
        return lock.LogicalLockResult(self.unlock, token)

    def unlock(self):
        self._lock.unlock()

    def break_lock(self):
        self._lock.break_lock()

    def save(self):
        with self.lock_write():
            self.save_without_locking()

    def save_without_locking(self):
        super().save()