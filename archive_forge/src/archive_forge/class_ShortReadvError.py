class ShortReadvError(PathError):
    _fmt = 'readv() read %(actual)s bytes rather than %(length)s bytes at %(offset)s for "%(path)s"%(extra)s'
    internal_error = True

    def __init__(self, path, offset, length, actual, extra=None):
        PathError.__init__(self, path, extra=extra)
        self.offset = offset
        self.length = length
        self.actual = actual