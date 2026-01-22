import os
import struct
from twisted.internet import fdesc
from twisted.internet.abstract import FileDescriptor
from twisted.python import _inotify, log
def _addWatch(self, path, mask, autoAdd, callbacks):
    """
        Private helper that abstracts the use of ctypes.

        Calls the internal inotify API and checks for any errors after the
        call. If there's an error L{INotify._addWatch} can raise an
        INotifyError. If there's no error it proceeds creating a watchpoint and
        adding a watchpath for inverse lookup of the file descriptor from the
        path.
        """
    path = path.asBytesMode()
    wd = self._inotify.add(self._fd, path, mask)
    iwp = _Watch(path, mask, autoAdd, callbacks)
    self._watchpoints[wd] = iwp
    self._watchpaths[path] = wd
    return wd