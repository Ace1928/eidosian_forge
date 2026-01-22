import errno
class ReadOnlyPool(ZFSError):
    errno = errno.EROFS
    message = 'Pool is read-only'

    def __init__(self, name):
        self.name = name