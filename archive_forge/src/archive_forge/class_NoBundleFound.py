class NoBundleFound(BzrError):
    _fmt = 'No bundle was found in "%(filename)s".'

    def __init__(self, filename):
        BzrError.__init__(self)
        self.filename = filename