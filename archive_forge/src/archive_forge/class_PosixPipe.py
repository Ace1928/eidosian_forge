import sys
import os
import socket
class PosixPipe:

    def __init__(self):
        self._rfd, self._wfd = os.pipe()
        self._set = False
        self._forever = False
        self._closed = False

    def close(self):
        os.close(self._rfd)
        os.close(self._wfd)
        self._closed = True

    def fileno(self):
        return self._rfd

    def clear(self):
        if not self._set or self._forever:
            return
        os.read(self._rfd, 1)
        self._set = False

    def set(self):
        if self._set or self._closed:
            return
        self._set = True
        os.write(self._wfd, b'*')

    def set_forever(self):
        self._forever = True
        self.set()