import errno
class StreamIOError(ZFSError):
    message = 'I/O error while writing or reading stream'

    def __init__(self, errno):
        self.errno = errno