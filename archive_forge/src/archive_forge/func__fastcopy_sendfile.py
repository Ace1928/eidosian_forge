import os
import sys
import stat
import fnmatch
import collections
import errno
def _fastcopy_sendfile(fsrc, fdst):
    """Copy data from one regular mmap-like fd to another by using
    high-performance sendfile(2) syscall.
    This should work on Linux >= 2.6.33 only.
    """
    global _USE_CP_SENDFILE
    try:
        infd = fsrc.fileno()
        outfd = fdst.fileno()
    except Exception as err:
        raise _GiveupOnFastCopy(err)
    try:
        blocksize = max(os.fstat(infd).st_size, 2 ** 23)
    except OSError:
        blocksize = 2 ** 27
    if sys.maxsize < 2 ** 32:
        blocksize = min(blocksize, 2 ** 30)
    offset = 0
    while True:
        try:
            sent = os.sendfile(outfd, infd, offset, blocksize)
        except OSError as err:
            err.filename = fsrc.name
            err.filename2 = fdst.name
            if err.errno == errno.ENOTSOCK:
                _USE_CP_SENDFILE = False
                raise _GiveupOnFastCopy(err)
            if err.errno == errno.ENOSPC:
                raise err from None
            if offset == 0 and os.lseek(outfd, 0, os.SEEK_CUR) == 0:
                raise _GiveupOnFastCopy(err)
            raise err
        else:
            if sent == 0:
                break
            offset += sent