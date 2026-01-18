import errno
import os
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
def writeToFD(fd, data):
    """
    Write data to file descriptor.

    Returns same thing FileDescriptor.writeSomeData would.

    @type fd: C{int}
    @param fd: non-blocking file descriptor to be written to.
    @type data: C{str} or C{buffer}
    @param data: bytes to write to fd.

    @return: number of bytes written, or CONNECTION_LOST.
    """
    try:
        return os.write(fd, data)
    except OSError as io:
        if io.errno in (errno.EAGAIN, errno.EINTR):
            return 0
        return CONNECTION_LOST