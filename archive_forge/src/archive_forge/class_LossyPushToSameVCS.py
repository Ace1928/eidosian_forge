class LossyPushToSameVCS(BzrError):
    _fmt = 'Lossy push not possible between %(source_branch)r and %(target_branch)r that are in the same VCS.'
    internal_error = True

    def __init__(self, source_branch, target_branch):
        self.source_branch = source_branch
        self.target_branch = target_branch