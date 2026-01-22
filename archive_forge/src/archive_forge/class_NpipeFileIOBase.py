import functools
import time
import io
import win32file
import win32pipe
import pywintypes
import win32event
import win32api
class NpipeFileIOBase(io.RawIOBase):

    def __init__(self, npipe_socket):
        self.sock = npipe_socket

    def close(self):
        super().close()
        self.sock = None

    def fileno(self):
        return self.sock.fileno()

    def isatty(self):
        return False

    def readable(self):
        return True

    def readinto(self, buf):
        return self.sock.recv_into(buf)

    def seekable(self):
        return False

    def writable(self):
        return False