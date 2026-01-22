import os
import sys
from mmap import mmap, ACCESS_READ
from mmap import ALLOCATIONGRANULARITY
class MapRegionList(list):
    """List of MapRegion instances associating a path with a list of regions."""
    __slots__ = ('_path_or_fd', '_file_size')

    def __new__(cls, path):
        return super().__new__(cls)

    def __init__(self, path_or_fd):
        self._path_or_fd = path_or_fd
        self._file_size = None

    def path_or_fd(self):
        """:return: path or file descriptor we are attached to"""
        return self._path_or_fd

    def file_size(self):
        """:return: size of file we manager"""
        if self._file_size is None:
            if isinstance(self._path_or_fd, str):
                self._file_size = os.stat(self._path_or_fd).st_size
            else:
                self._file_size = os.fstat(self._path_or_fd).st_size
        return self._file_size