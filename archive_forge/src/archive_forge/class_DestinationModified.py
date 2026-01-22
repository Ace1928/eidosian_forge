import errno
class DestinationModified(ZFSError):
    errno = errno.ETXTBSY
    message = 'Destination dataset has modifications that can not be undone'

    def __init__(self, name):
        self.name = name