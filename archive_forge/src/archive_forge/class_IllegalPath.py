class IllegalPath(BzrError):
    _fmt = 'The path %(path)s is not permitted on this platform'

    def __init__(self, path):
        BzrError.__init__(self)
        self.path = path