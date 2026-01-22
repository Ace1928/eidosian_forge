class NoRoundtrippingSupport(BzrError):
    _fmt = 'Roundtripping is not supported between %(source_branch)r and %(target_branch)r.'
    internal_error = True

    def __init__(self, source_branch, target_branch):
        self.source_branch = source_branch
        self.target_branch = target_branch