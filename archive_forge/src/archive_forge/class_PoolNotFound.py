import errno
class PoolNotFound(ZFSError):
    errno = errno.EXDEV
    message = 'No such pool'

    def __init__(self, name):
        self.name = name