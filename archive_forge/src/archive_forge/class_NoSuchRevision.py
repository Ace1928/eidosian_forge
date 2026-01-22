class NoSuchRevision(InternalBzrError):
    revision: bytes
    _fmt = '%(branch)s has no revision %(revision)s'

    def __init__(self, branch, revision):
        BzrError.__init__(self, branch=branch, revision=revision)