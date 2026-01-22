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
class LockableConfig(IniBasedConfig):
    """A configuration needing explicit locking for access.

    If several processes try to write the config file, the accesses need to be
    serialized.

    Daughter classes should use the self.lock_write() decorator method when
    they upate a config (they call, directly or indirectly, the
    ``_write_config_file()`` method. These methods (typically ``set_option()``
    and variants must reload the config file from disk before calling
    ``_write_config_file()``), this can be achieved by calling the
    ``self.reload()`` method. Note that the lock scope should cover both the
    reading and the writing of the config file which is why the decorator can't
    be applied to ``_write_config_file()`` only.

    This should be enough to implement the following logic:
    - lock for exclusive write access,
    - reload the config file from disk,
    - set the new value
    - unlock

    This logic guarantees that a writer can update a value without erasing an
    update made by another writer.
    """
    lock_name = 'lock'

    def __init__(self, file_name):
        super().__init__(file_name=file_name)
        self.dir = osutils.dirname(osutils.safe_unicode(self.file_name))
        self.transport = transport.get_transport_from_path(self.dir)
        self._lock = lockdir.LockDir(self.transport, self.lock_name)

    def _create_from_string(self, unicode_bytes, save):
        super()._create_from_string(unicode_bytes, False)
        if save:
            self.lock_write()
            self._write_config_file()
            self.unlock()

    def lock_write(self, token=None):
        """Takes a write lock in the directory containing the config file.

        If the directory doesn't exist it is created.
        """
        bedding.ensure_config_dir_exists(self.dir)
        token = self._lock.lock_write(token)
        return lock.LogicalLockResult(self.unlock, token)

    def unlock(self):
        self._lock.unlock()

    def break_lock(self):
        self._lock.break_lock()

    def remove_user_option(self, option_name, section_name=None):
        with self.lock_write():
            super().remove_user_option(option_name, section_name)

    def _write_config_file(self):
        if self._lock is None or not self._lock.is_held:
            raise errors.ObjectNotLocked(self)
        super()._write_config_file()