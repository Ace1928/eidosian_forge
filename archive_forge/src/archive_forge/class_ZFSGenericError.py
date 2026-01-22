import errno
class ZFSGenericError(ZFSError):

    def __init__(self, errno, name, message):
        self.errno = errno
        self.message = message
        self.name = name