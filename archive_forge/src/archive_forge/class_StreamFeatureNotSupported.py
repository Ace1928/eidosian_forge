import errno
class StreamFeatureNotSupported(ZFSError):
    errno = errno.ENOTSUP
    message = 'Stream contains unsupported feature'