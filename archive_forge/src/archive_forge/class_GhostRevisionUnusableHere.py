class GhostRevisionUnusableHere(BzrError):
    _fmt = 'Ghost revision {%(revision_id)s} cannot be used here.'

    def __init__(self, revision_id):
        BzrError.__init__(self)
        self.revision_id = revision_id