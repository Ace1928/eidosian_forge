import errno
class BadHoldCleanupFD(ZFSError):
    errno = errno.EBADF
    message = 'Bad file descriptor as cleanup file descriptor'