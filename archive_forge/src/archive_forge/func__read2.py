import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def _read2(self, n):
    if self._compress_left <= 0:
        return b''
    n = max(n, self.MIN_READ_SIZE)
    n = min(n, self._compress_left)
    data = self._fileobj.read(n)
    self._compress_left -= len(data)
    if not data:
        raise EOFError
    if self._decrypter is not None:
        data = self._decrypter(data)
    return data