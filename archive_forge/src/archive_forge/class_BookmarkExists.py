import errno
class BookmarkExists(ZFSError):
    errno = errno.EEXIST
    message = 'Bookmark already exists'

    def __init__(self, name):
        self.name = name