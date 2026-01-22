import errno
class SuspendedPool(ZFSError):
    errno = errno.EAGAIN
    message = 'Pool is suspended'

    def __init__(self, name):
        self.name = name