import errno
class ZIOError(ZFSError):
    errno = errno.EIO
    message = 'I/O error'

    def __init__(self, name):
        self.name = name