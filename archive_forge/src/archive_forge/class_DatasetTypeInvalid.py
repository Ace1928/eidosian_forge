import errno
class DatasetTypeInvalid(ZFSError):
    errno = errno.EINVAL
    message = 'Specified dataset type is unknown'

    def __init__(self, name):
        self.name = name