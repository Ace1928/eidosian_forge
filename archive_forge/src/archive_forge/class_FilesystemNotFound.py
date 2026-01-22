import errno
class FilesystemNotFound(DatasetNotFound):
    message = 'Filesystem not found'

    def __init__(self, name):
        self.name = name