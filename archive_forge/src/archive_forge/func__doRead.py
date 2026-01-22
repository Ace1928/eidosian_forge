import os
import struct
from twisted.internet import fdesc
from twisted.internet.abstract import FileDescriptor
from twisted.python import _inotify, log
def _doRead(self, in_):
    """
        Work on the data just read from the file descriptor.
        """
    self._buffer += in_
    while len(self._buffer) >= 16:
        wd, mask, cookie, size = struct.unpack('=LLLL', self._buffer[0:16])
        if size:
            name = self._buffer[16:16 + size].rstrip(b'\x00')
        else:
            name = None
        self._buffer = self._buffer[16 + size:]
        try:
            iwp = self._watchpoints[wd]
        except KeyError:
            continue
        path = iwp.path.asBytesMode()
        if name:
            path = path.child(name)
        iwp._notify(path, mask)
        if iwp.autoAdd and mask & IN_ISDIR and mask & IN_CREATE:
            new_wd = self.watch(path, mask=iwp.mask, autoAdd=True, callbacks=iwp.callbacks)
            self.reactor.callLater(0, self._addChildren, self._watchpoints[new_wd])
        if mask & IN_DELETE_SELF:
            self._rmWatch(wd)
            self.loseConnection()