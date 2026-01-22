import errno
class DatasetExists(ZFSError):
    """
    This exception is raised when an operation failure can be caused by an existing
    snapshot or filesystem and it is impossible to distinguish between
    the causes.
    """
    errno = errno.EEXIST
    message = 'Dataset already exists'

    def __init__(self, name):
        self.name = name