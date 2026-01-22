import errno
class BadStream(ZFSError):
    errno = errno.EINVAL
    message = 'Bad backup stream'