import errno
class WrongParent(ZFSError):
    errno = errno.EINVAL
    message = 'Parent dataset is not a filesystem'

    def __init__(self, name):
        self.name = name