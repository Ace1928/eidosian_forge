import os
import sys
from mmap import mmap, ACCESS_READ
from mmap import ALLOCATIONGRANULARITY
class MapRegion:
    """Defines a mapped region of memory, aligned to pagesizes

    **Note:** deallocates used region automatically on destruction"""
    __slots__ = ['_b', '_mf', '_uc', '_size', '__weakref__']

    def __init__(self, path_or_fd, ofs, size, flags=0):
        """Initialize a region, allocate the memory map
        :param path_or_fd: path to the file to map, or the opened file descriptor
        :param ofs: **aligned** offset into the file to be mapped
        :param size: if size is larger then the file on disk, the whole file will be
            allocated the the size automatically adjusted
        :param flags: additional flags to be given when opening the file.
        :raise Exception: if no memory can be allocated"""
        self._b = ofs
        self._size = 0
        self._uc = 0
        if isinstance(path_or_fd, int):
            fd = path_or_fd
        else:
            fd = os.open(path_or_fd, os.O_RDONLY | getattr(os, 'O_BINARY', 0) | flags)
        try:
            kwargs = dict(access=ACCESS_READ, offset=ofs)
            corrected_size = size
            sizeofs = ofs
            actual_size = min(os.fstat(fd).st_size - sizeofs, corrected_size)
            self._mf = mmap(fd, actual_size, **kwargs)
            self._size = len(self._mf)
        finally:
            if isinstance(path_or_fd, str):
                os.close(fd)
        self.increment_client_count()

    def __repr__(self):
        return 'MapRegion<%i, %i>' % (self._b, self.size())

    def buffer(self):
        """:return: a buffer containing the memory"""
        return self._mf

    def map(self):
        """:return: a memory map containing the memory"""
        return self._mf

    def ofs_begin(self):
        """:return: absolute byte offset to the first byte of the mapping"""
        return self._b

    def size(self):
        """:return: total size of the mapped region in bytes"""
        return self._size

    def ofs_end(self):
        """:return: Absolute offset to one byte beyond the mapping into the file"""
        return self._b + self._size

    def includes_ofs(self, ofs):
        """:return: True if the given offset can be read in our mapped region"""
        return self._b <= ofs < self._b + self._size

    def client_count(self):
        """:return: number of clients currently using this region"""
        return self._uc

    def increment_client_count(self, ofs=1):
        """Adjust the usage count by the given positive or negative offset.
        If usage count equals 0, we will auto-release our resources
        :return: True if we released resources, False otherwise. In the latter case, we can still be used"""
        self._uc += ofs
        assert self._uc > -1, 'Increments must match decrements, usage counter negative: %i' % self._uc
        if self.client_count() == 0:
            self.release()
            return True
        else:
            return False

    def release(self):
        """Release all resources this instance might hold. Must only be called if there usage_count() is zero"""
        self._mf.close()