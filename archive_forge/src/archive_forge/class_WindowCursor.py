from .util import (
import sys
from functools import reduce
class WindowCursor:
    """
    Pointer into the mapped region of the memory manager, keeping the map
    alive until it is destroyed and no other client uses it.

    Cursors should not be created manually, but are instead returned by the SlidingWindowMapManager

    **Note:**: The current implementation is suited for static and sliding window managers, but it also means
    that it must be suited for the somewhat quite different sliding manager. It could be improved, but
    I see no real need to do so."""
    __slots__ = ('_manager', '_rlist', '_region', '_ofs', '_size')

    def __init__(self, manager=None, regions=None):
        self._manager = manager
        self._rlist = regions
        self._region = None
        self._ofs = 0
        self._size = 0

    def __del__(self):
        self._destroy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._destroy()

    def _destroy(self):
        """Destruction code to decrement counters"""
        self.unuse_region()
        if self._rlist is not None:
            try:
                if len(self._rlist) == 0:
                    self._manager._fdict.pop(self._rlist.path_or_fd())
            except (TypeError, KeyError):
                pass

    def _copy_from(self, rhs):
        """Copy all data from rhs into this instance, handles usage count"""
        self._manager = rhs._manager
        self._rlist = type(rhs._rlist)(rhs._rlist)
        self._region = rhs._region
        self._ofs = rhs._ofs
        self._size = rhs._size
        for region in self._rlist:
            region.increment_client_count()
        if self._region is not None:
            self._region.increment_client_count()

    def __copy__(self):
        """copy module interface"""
        cpy = type(self)()
        cpy._copy_from(self)
        return cpy

    def assign(self, rhs):
        """Assign rhs to this instance. This is required in order to get a real copy.
        Alternatively, you can copy an existing instance using the copy module"""
        self._destroy()
        self._copy_from(rhs)

    def use_region(self, offset=0, size=0, flags=0):
        """Assure we point to a window which allows access to the given offset into the file

        :param offset: absolute offset in bytes into the file
        :param size: amount of bytes to map. If 0, all available bytes will be mapped
        :param flags: additional flags to be given to os.open in case a file handle is initially opened
            for mapping. Has no effect if a region can actually be reused.
        :return: this instance - it should be queried for whether it points to a valid memory region.
            This is not the case if the mapping failed because we reached the end of the file

        **Note:**: The size actually mapped may be smaller than the given size. If that is the case,
        either the file has reached its end, or the map was created between two existing regions"""
        need_region = True
        man = self._manager
        fsize = self._rlist.file_size()
        size = min(size or fsize, man.window_size() or fsize)
        if self._region is not None:
            if self._region.includes_ofs(offset):
                need_region = False
            else:
                self.unuse_region()
        if offset >= fsize:
            return self
        if need_region:
            self._region = man._obtain_region(self._rlist, offset, size, flags, False)
            self._region.increment_client_count()
        self._ofs = offset - self._region._b
        self._size = min(size, self._region.ofs_end() - offset)
        return self

    def unuse_region(self):
        """Unuse the current region. Does nothing if we have no current region

        **Note:** the cursor unuses the region automatically upon destruction. It is recommended
        to un-use the region once you are done reading from it in persistent cursors as it
        helps to free up resource more quickly"""
        if self._region is not None:
            self._region.increment_client_count(-1)
        self._region = None

    def buffer(self):
        """Return a buffer object which allows access to our memory region from our offset
        to the window size. Please note that it might be smaller than you requested when calling use_region()

        **Note:** You can only obtain a buffer if this instance is_valid() !

        **Note:** buffers should not be cached passed the duration of your access as it will
        prevent resources from being freed even though they might not be accounted for anymore !"""
        return memoryview(self._region.buffer())[self._ofs:self._ofs + self._size]

    def map(self):
        """
        :return: the underlying raw memory map. Please not that the offset and size is likely to be different
            to what you set as offset and size. Use it only if you are sure about the region it maps, which is the whole
            file in case of StaticWindowMapManager"""
        return self._region.map()

    def is_valid(self):
        """:return: True if we have a valid and usable region"""
        return self._region is not None

    def is_associated(self):
        """:return: True if we are associated with a specific file already"""
        return self._rlist is not None

    def ofs_begin(self):
        """:return: offset to the first byte pointed to by our cursor

        **Note:** only if is_valid() is True"""
        return self._region._b + self._ofs

    def ofs_end(self):
        """:return: offset to one past the last available byte"""
        return self._region._b + self._ofs + self._size

    def size(self):
        """:return: amount of bytes we point to"""
        return self._size

    def region(self):
        """:return: our mapped region, or None if nothing is mapped yet
        :raise AssertionError: if we have no current region. This is only useful for debugging"""
        return self._region

    def includes_ofs(self, ofs):
        """:return: True if the given absolute offset is contained in the cursors
            current region

        **Note:** cursor must be valid for this to work"""
        return self._region._b + self._ofs <= ofs < self._region._b + self._ofs + self._size

    def file_size(self):
        """:return: size of the underlying file"""
        return self._rlist.file_size()

    def path_or_fd(self):
        """:return: path or file descriptor of the underlying mapped file"""
        return self._rlist.path_or_fd()

    def path(self):
        """:return: path of the underlying mapped file
        :raise ValueError: if attached path is not a path"""
        if isinstance(self._rlist.path_or_fd(), int):
            raise ValueError('Path queried although mapping was applied to a file descriptor')
        return self._rlist.path_or_fd()

    def fd(self):
        """:return: file descriptor used to create the underlying mapping.

        **Note:** it is not required to be valid anymore
        :raise ValueError: if the mapping was not created by a file descriptor"""
        if isinstance(self._rlist.path_or_fd(), str):
            raise ValueError('File descriptor queried although mapping was generated from path')
        return self._rlist.path_or_fd()