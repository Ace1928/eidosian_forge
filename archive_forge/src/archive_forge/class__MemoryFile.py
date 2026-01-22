from __future__ import absolute_import, unicode_literals
import typing
import contextlib
import io
import os
import six
import time
from collections import OrderedDict
from threading import RLock
from . import errors
from ._typing import overload
from .base import FS
from .copy import copy_modified_time
from .enums import ResourceType, Seek
from .info import Info
from .mode import Mode
from .path import iteratepath, normpath, split
@six.python_2_unicode_compatible
class _MemoryFile(io.RawIOBase):

    def __init__(self, path, memory_fs, mode, dir_entry):
        super(_MemoryFile, self).__init__()
        self._path = path
        self._memory_fs = memory_fs
        self._mode = Mode(mode)
        self._dir_entry = dir_entry
        self._bytes_io = typing.cast(io.BytesIO, dir_entry.bytes_file)
        self.accessed_time = time.time()
        self.modified_time = time.time()
        self.pos = 0
        if self._mode.truncate:
            with self._dir_entry.lock:
                self._bytes_io.seek(0)
                self._bytes_io.truncate()
        elif self._mode.appending:
            with self._dir_entry.lock:
                self._bytes_io.seek(0, os.SEEK_END)
                self.pos = self._bytes_io.tell()

    def __str__(self):
        _template = "<memoryfile '{path}' '{mode}'>"
        return _template.format(path=self._path, mode=self._mode)

    @property
    def mode(self):
        return self._mode.to_platform_bin()

    @contextlib.contextmanager
    def _seek_lock(self):
        with self._dir_entry.lock:
            self._bytes_io.seek(self.pos)
            yield
            self.pos = self._bytes_io.tell()

    def on_modify(self):
        """Called when file data is modified."""
        self._dir_entry.modified_time = self.modified_time = time.time()

    def on_access(self):
        """Called when file is accessed."""
        self._dir_entry.accessed_time = self.accessed_time = time.time()

    def flush(self):
        pass

    def __iter__(self):
        self._bytes_io.seek(self.pos)
        for line in self._bytes_io:
            yield line

    def next(self):
        with self._seek_lock():
            self.on_access()
            return next(self._bytes_io)
    __next__ = next

    def readline(self, size=None):
        if not self._mode.reading:
            raise IOError('File not open for reading')
        with self._seek_lock():
            self.on_access()
            return self._bytes_io.readline(size)

    def close(self):
        if not self.closed:
            with self._dir_entry.lock:
                self._dir_entry.remove_open_file(self)
                super(_MemoryFile, self).close()

    def read(self, size=None):
        if not self._mode.reading:
            raise IOError('File not open for reading')
        with self._seek_lock():
            self.on_access()
            return self._bytes_io.read(size)

    def readable(self):
        return self._mode.reading

    def readinto(self, buffer):
        if not self._mode.reading:
            raise IOError('File not open for reading')
        with self._seek_lock():
            self.on_access()
            return self._bytes_io.readinto(buffer)

    def readlines(self, hint=-1):
        if not self._mode.reading:
            raise IOError('File not open for reading')
        with self._seek_lock():
            self.on_access()
            return self._bytes_io.readlines(hint)

    def seekable(self):
        return True

    def seek(self, pos, whence=Seek.set):
        with self._seek_lock():
            self.on_access()
            return self._bytes_io.seek(pos, int(whence))

    def tell(self):
        return self.pos

    def truncate(self, size=None):
        with self._seek_lock():
            self.on_modify()
            new_size = self._bytes_io.truncate(size)
            if size is not None and self._bytes_io.tell() < size:
                file_size = self._bytes_io.seek(0, os.SEEK_END)
                self._bytes_io.write(b'\x00' * (size - file_size))
                self._bytes_io.seek(-size + file_size, os.SEEK_END)
            return size or new_size

    def writable(self):
        return self._mode.writing

    def write(self, data):
        if not self._mode.writing:
            raise IOError('File not open for writing')
        with self._seek_lock():
            self.on_modify()
            return self._bytes_io.write(data)

    def writelines(self, sequence):
        with self._seek_lock():
            self.on_modify()
            self._bytes_io.writelines(sequence)