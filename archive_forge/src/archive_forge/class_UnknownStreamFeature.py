import errno
class UnknownStreamFeature(ZFSError):
    errno = errno.ENOTSUP
    message = 'Unknown feature requested for stream'