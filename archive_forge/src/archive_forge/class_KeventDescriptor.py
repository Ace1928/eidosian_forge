from __future__ import with_statement
from wandb_watchdog.utils import platform
import threading
import errno
import sys
import stat
import os
from wandb_watchdog.observers.api import (
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot
from wandb_watchdog.events import (
class KeventDescriptor(object):
    """
    A kevent descriptor convenience data structure to keep together:

        * kevent
        * directory status
        * path
        * file descriptor

    :param path:
        Path string for which a kevent descriptor will be created.
    :param is_directory:
        ``True`` if the path refers to a directory; ``False`` otherwise.
    :type is_directory:
        ``bool``
    """

    def __init__(self, path, is_directory):
        self._path = absolute_path(path)
        self._is_directory = is_directory
        self._fd = os.open(path, WATCHDOG_OS_OPEN_FLAGS)
        self._kev = select.kevent(self._fd, filter=WATCHDOG_KQ_FILTER, flags=WATCHDOG_KQ_EV_FLAGS, fflags=WATCHDOG_KQ_FFLAGS)

    @property
    def fd(self):
        """OS file descriptor for the kevent descriptor."""
        return self._fd

    @property
    def path(self):
        """The path associated with the kevent descriptor."""
        return self._path

    @property
    def kevent(self):
        """The kevent object associated with the kevent descriptor."""
        return self._kev

    @property
    def is_directory(self):
        """Determines whether the kevent descriptor refers to a directory.

        :returns:
            ``True`` or ``False``
        """
        return self._is_directory

    def close(self):
        """
        Closes the file descriptor associated with a kevent descriptor.
        """
        try:
            os.close(self.fd)
        except OSError:
            pass

    @property
    def key(self):
        return (self.path, self.is_directory)

    def __eq__(self, descriptor):
        return self.key == descriptor.key

    def __ne__(self, descriptor):
        return self.key != descriptor.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return '<KeventDescriptor: path=%s, is_directory=%s>' % (self.path, self.is_directory)