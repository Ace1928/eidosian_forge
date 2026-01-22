class InvalidRevisionId(BzrError):
    _fmt = 'Invalid revision-id {%(revision_id)s} in %(branch)s'

    def __init__(self, revision_id, branch):
        BzrError.__init__(self)
        self.revision_id = revision_id
        self.branch = branch