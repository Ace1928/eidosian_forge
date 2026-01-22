class PathsDoNotExist(BzrError):
    _fmt = 'Path(s) do not exist: %(paths_as_string)s%(extra)s'

    def __init__(self, paths, extra=None):
        from breezy.osutils import quotefn
        BzrError.__init__(self)
        self.paths = paths
        self.paths_as_string = ' '.join([quotefn(p) for p in paths])
        if extra:
            self.extra = ': ' + str(extra)
        else:
            self.extra = ''