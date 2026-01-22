import errno
class DatasetBusy(ZFSError):
    errno = errno.EBUSY
    message = 'Dataset is busy'

    def __init__(self, name):
        self.name = name