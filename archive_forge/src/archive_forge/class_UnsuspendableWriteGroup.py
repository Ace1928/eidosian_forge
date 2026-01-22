class UnsuspendableWriteGroup(BzrError):
    _fmt = 'Repository %(repository)s cannot suspend a write group.'
    internal_error = True

    def __init__(self, repository):
        self.repository = repository