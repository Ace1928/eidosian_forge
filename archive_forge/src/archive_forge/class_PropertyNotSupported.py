import errno
class PropertyNotSupported(ZFSError):
    errno = errno.ENOTSUP
    message = 'Property is not supported in this version'

    def __init__(self, name):
        self.name = name