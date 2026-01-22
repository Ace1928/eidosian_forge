class PathsNotVersionedError(BzrError):
    """Used when reporting several paths which are not versioned"""
    _fmt = 'Path(s) are not versioned: %(paths_as_string)s'

    def __init__(self, paths):
        from breezy.osutils import quotefn
        BzrError.__init__(self)
        self.paths = paths
        self.paths_as_string = ' '.join([quotefn(p) for p in paths])