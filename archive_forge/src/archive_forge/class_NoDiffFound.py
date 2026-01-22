class NoDiffFound(BzrError):
    _fmt = 'Could not find an appropriate Differ for file "%(path)s"'

    def __init__(self, path):
        BzrError.__init__(self, path)