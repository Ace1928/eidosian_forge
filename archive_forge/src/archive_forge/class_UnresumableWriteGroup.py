class UnresumableWriteGroup(BzrError):
    _fmt = 'Repository %(repository)s cannot resume write group %(write_groups)r: %(reason)s'
    internal_error = True

    def __init__(self, repository, write_groups, reason):
        self.repository = repository
        self.write_groups = write_groups
        self.reason = reason