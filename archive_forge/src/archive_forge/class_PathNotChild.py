class PathNotChild(PathError):
    _fmt = 'Path "%(path)s" is not a child of path "%(base)s"%(extra)s'
    internal_error = False

    def __init__(self, path, base, extra=None):
        BzrError.__init__(self)
        self.path = path
        self.base = base
        if extra:
            self.extra = ': ' + str(extra)
        else:
            self.extra = ''