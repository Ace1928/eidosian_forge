import sys
import os
import socket
class WindowsPipe:
    """
    On Windows, only an OS-level "WinSock" may be used in select(), but reads
    and writes must be to the actual socket object.
    """

    def __init__(self):
        serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serv.bind(('127.0.0.1', 0))
        serv.listen(1)
        self._rsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._rsock.connect(('127.0.0.1', serv.getsockname()[1]))
        self._wsock, addr = serv.accept()
        serv.close()
        self._set = False
        self._forever = False
        self._closed = False

    def close(self):
        self._rsock.close()
        self._wsock.close()
        self._closed = True

    def fileno(self):
        return self._rsock.fileno()

    def clear(self):
        if not self._set or self._forever:
            return
        self._rsock.recv(1)
        self._set = False

    def set(self):
        if self._set or self._closed:
            return
        self._set = True
        self._wsock.send(b'*')

    def set_forever(self):
        self._forever = True
        self.set()