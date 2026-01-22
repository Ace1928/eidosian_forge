class TargetNotBranch(BzrError):
    """A merge directive's target branch is required, but isn't a branch"""
    _fmt = 'Your branch does not have all of the revisions required in order to merge this merge directive and the target location specified in the merge directive is not a branch: %(location)s.'

    def __init__(self, location):
        BzrError.__init__(self)
        self.location = location