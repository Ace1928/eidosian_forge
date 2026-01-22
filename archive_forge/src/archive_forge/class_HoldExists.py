import errno
class HoldExists(ZFSError):
    errno = errno.EEXIST
    message = 'Hold with a given tag already exists on snapshot'

    def __init__(self, name):
        self.name = name