from .util import (
import sys
from functools import reduce
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